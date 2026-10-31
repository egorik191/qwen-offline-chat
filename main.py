# -*- coding: utf-8 -*-
"""
Egor AI — офлайн-чат с локальной нейросетью (Qwen2.5-3B-Instruct).
Kivy + pylolm (llama.cpp). Полностью офлайн.
"""

import os
import re
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "qwen2.5-3b-instruct-q4_k_m.gguf")

os.environ.setdefault("PYLOLM_LIB_DIR", os.path.join(BASE_DIR, "pylolm_libs"))
os.environ.setdefault("PYLOLM_AUTO_DOWNLOAD", "0")

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, RoundedRectangle

from pylolm import Llama, PyLoLmError  # noqa: E402

Window.soft_input_mode = "below_target"

# ------- палитра в духе Claude (тёмная тёплая)
BG = "#1f1e1d"        # фон
PANEL = "#2e2c29"     # панели
CARD = "#35322e"      # чипы, кнопки
ACCENT = "#d97757"    # терракота Claude
MY_BUBBLE = "#c96442" # пузырь пользователя
AI_BUBBLE = "#2c2a27" # пузырь ассистента
TEXT_MAIN = "#f2eee7"
TEXT_SOFT = "#a8a29b"

SYSTEM_PROMPT = ("Ты — полезный ассистент по имени Egor AI. Отвечай кратко и "
                 "по-русски, если вопрос на русском. Если просят код — пиши код.")

TOOLS_PROMPT = """

Ты можешь работать с файлами устройства. Доступные команды (пиши их РОВНО в таком виде в ответе, по одной за раз, и останавливайся — результат тебе пришлют):
[CMD:список путь] — показать файлы папки
[CMD:чтение путь] — прочитать файл
[CMD:запись путь]
текст файла
[/CMD] — создать/перезаписать файл
[CMD:удалить путь] — удалить файл
[CMD:папка путь] — создать папку
Пример: пользователь просит сохранить заметку — пиши [CMD:запись /sdcard/Documents/заметка.txt]\nтекст\n[/CMD] и жди результат. После результата продолжай ответ."""
MAX_TOKENS = 700
MAX_CONTINUATIONS = 4
MAX_AGENT_STEPS = 6      # максимум команд за один запрос
N_CTX = 2048

TOOLS_HELP = ("Режим файлов включён. Могу: список/чтение/запись/удаление "
              "файлов и создание папок — в папке приложения и на памяти "
              "телефона (/sdcard), если Android разрешит.")


# ---------------------------------------------------------------- инструменты

def allowed_roots():
    """Папки, внутри которых агенту разрешено работать."""
    roots = [BASE_DIR]                       # папка приложения
    for p in ("/sdcard", os.path.expanduser("~")):
        if os.path.isdir(p):
            roots.append(p)
    return roots


def safe_path(path):
    """Проверить путь: только внутри разрешённых корней."""
    p = os.path.realpath(os.path.expanduser(str(path).strip()))
    for root in allowed_roots():
        if p == root or p.startswith(root + os.sep):
            return p
    return None


def tool_list(path="."):
    p = safe_path(path)
    if p is None:
        return "ОШИБКА: путь вне разрешённых папок"
    if not os.path.isdir(p):
        return f"ОШИБКА: папка не найдена: {path}"
    items = []
    for name in sorted(os.listdir(p))[:50]:
        full = os.path.join(p, name)
        items.append(("[папка] " if os.path.isdir(full) else "") + name)
    return f"Папка {p}:\n" + ("\n".join(items) or "(пусто)")


def tool_read(path):
    p = safe_path(path)
    if p is None:
        return "ОШИБКА: путь вне разрешённых папок"
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            data = f.read(4000)
        return f"Содержимое {p}:\n{data}"
    except OSError as exc:
        return f"ОШИБКА чтения: {exc}"


def tool_write(path, content):
    p = safe_path(path)
    if p is None:
        return "ОШИБКА: путь вне разрешённых папок"
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Записано {len(content)} символов в {p}"
    except OSError as exc:
        if "Permission" in type(exc).__name__:
            return ("ОШИБКА: Android запретил запись. Дай приложению "
                    "'Все файлы': Настройки → Приложения → Egor AI → "
                    "Разрешения → Файлы и медиа → Разрешить доступ ко всем файлам")
        return f"ОШИБКА записи: {exc}"


def tool_delete(path):
    p = safe_path(path)
    if p is None:
        return "ОШИБКА: путь вне разрешённых папок"
    try:
        if os.path.isdir(p):
            os.rmdir(p)
            return f"Папка удалена: {p}"
        os.remove(p)
        return f"Файл удалён: {p}"
    except OSError as exc:
        return f"ОШИБКА удаления: {exc}"


def tool_mkdir(path):
    p = safe_path(path)
    if p is None:
        return "ОШИБКА: путь вне разрешённых папок"
    try:
        os.makedirs(p, exist_ok=True)
        return f"Папка создана: {p}"
    except OSError as exc:
        return f"ОШИБКА: {exc}"


def run_tool(cmd_line):
    """Выполнить [CMD:...]; вернуть результат строкой для модели."""
    m = re.match(r"\[CMD:([а-яё]+)\s+([^\]]+)\]?", cmd_line, re.I)
    if not m:
        return "ОШИБКА: не понял команду"
    kind, path = m.group(1).lower(), m.group(2).strip()
    if kind == "список":
        return tool_list(path)
    if kind == "чтение":
        return tool_read(path)
    if kind == "запись":
        body = cmd_line.split("]", 1)[1]
        body = body.split("[/CMD]")[0].strip("\n")
        return tool_write(path, body)
    if kind == "удалить":
        return tool_delete(path)
    if kind == "папка":
        return tool_mkdir(path)
    return f"ОШИБКА: неизвестная команда '{kind}'"


def hex_color(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [1]


class PillButton(Button):
    """Скруглённая кнопка-пилюля (без override on_press — только canvas)."""

    def __init__(self, text, bg=CARD, fg=TEXT_MAIN, font=14, radius=20,
                 **kw):
        super().__init__(**kw)
        self.text = text
        self._bg = hex_color(bg)
        self._radius = [dp(radius)]
        self.font_size = dp(font)
        self.color = hex_color(fg)
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)  # рисуем сами
        with self.canvas.before:
            self._color = Color(*self._bg)
            self._rect = RoundedRectangle(pos=self.pos, size=self.size,
                                          radius=self._radius)
        self.bind(pos=self._upd, size=self._upd, disabled=self._upd,
                  state=self._upd)

    def _upd(self, *_):
        self._rect.pos = self.pos
        self._rect.size = self.size
        # цвет меняем у инструкции Color, у RoundedRectangle нельзя
        c = self._bg
        k = 0.8 if self.state == "down" else 1.0
        a = 1.0 if not self.disabled else 0.4
        self._color.rgba = (c[0] * k, c[1] * k, c[2] * k, a)


class CodeCard(BoxLayout):
    """Тёмная карточка с кодом и кнопкой «Копировать»."""

    def __init__(self, code, lang="", **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.size_hint = (1, None)
        self.padding = [dp(12), dp(8)]
        self.spacing = dp(4)
        self.bind(minimum_height=self.setter("height"))
        self.code = code

        head = BoxLayout(size_hint_y=None, height=dp(28))
        lang_lbl = Label(text=lang or "код", font_size=dp(11),
                         color=hex_color("#8f897f"), halign="left")
        lang_lbl.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
        btn = PillButton("Копировать", bg="#4a453e", fg=TEXT_MAIN, font=11,
                         radius=14, size_hint_x=None, width=dp(120),
                         height=dp(28))
        btn.bind(on_press=lambda _: self.copy())
        head.add_widget(lang_lbl)
        head.add_widget(btn)

        body = Label(text=code.strip("\n"), font_size=dp(13),
                     color=hex_color("#efe9df"), halign="left", valign="top",
                     size_hint_y=None)
        body.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
        body.bind(texture_size=lambda i, s: setattr(i, "height", s[1]))

        self.add_widget(head)
        self.add_widget(body)

        with self.canvas.before:
            self._color = Color(*hex_color("#161411"))
            self._rect = RoundedRectangle(pos=self.pos, size=self.size,
                                          radius=[dp(12)])
        self.bind(pos=lambda _, p: setattr(self._rect, "pos", p))
        self.bind(size=lambda _, s: setattr(self._rect, "size", s))

    def copy(self):
        from kivy.core.clipboard import Clipboard
        Clipboard.copy(self.code)
        Clock.schedule_once(lambda dt: self._flash_done())

    def _flash_done(self):
        # лёгкая обратная связь: карточка вспыхивает тёплым
        self._color.rgba = hex_color("#33261a")


class Bubble(BoxLayout):
    """Сообщение: имя, текст, время; высота следует за содержимым."""

    def __init__(self, role, text="", **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.size_hint = (0.86, None)
        self.padding = [dp(14), dp(10)]
        self.spacing = dp(2)
        self.bind(minimum_height=self.setter("height"))

        name, bg, nc = {
            "user": ("Ты", MY_BUBBLE, "#f7e7dd"),
            "assistant": ("Egor AI", AI_BUBBLE, "#d9a583"),
            "system": ("", "#3a332a", TEXT_SOFT),
        }[role]

        if name:
            head = Label(text=name, font_size=dp(11), bold=True,
                         color=hex_color(nc), size_hint_y=None, height=dp(16),
                         halign="left")
            head.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
            self.add_widget(head)

        self.label = Label(text=text, font_size=dp(15),
                           color=hex_color(TEXT_MAIN), halign="left",
                           valign="top", size_hint_y=None)
        self.label.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
        self.label.bind(
            texture_size=lambda i, s: setattr(i, "height", s[1]))
        self.add_widget(self.label)

        if role != "system":
            self.time_lbl = Label(
                text=time.strftime("%H:%M"), font_size=dp(10),
                color=hex_color("#8f897f"), size_hint_y=None,
                height=dp(13), halign="right")
            self.time_lbl.bind(
                width=lambda i, w: setattr(i, "text_size", (w, None)))
            self.add_widget(self.time_lbl)

        with self.canvas.before:
            Color(*hex_color(bg))
            self._rect = RoundedRectangle(pos=self.pos, size=self.size,
                                          radius=[dp(18)])
        self.bind(pos=lambda _, p: setattr(self._rect, "pos", p))
        self.bind(size=lambda _, s: setattr(self._rect, "size", s))

    def format_with_code(self, text):
        """Переформатировать пузырь: блоки ``` становятся карточками
        с кнопкой «Копировать». Вызывается после завершения ответа."""
        self.clear_widgets()
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:  # нечётные части — код
                lines = part.split("\n")
                lang = lines[0].strip() if lines and len(lines) > 1 else ""
                code = "\n".join(lines[1:]) if lang else part
                self.add_widget(CodeCard(code, lang))
            elif part.strip():
                lbl = Label(text=part.strip("\n"), font_size=dp(15),
                            color=hex_color(TEXT_MAIN), halign="left",
                            valign="top", size_hint_y=None)
                lbl.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
                lbl.bind(texture_size=lambda i, s: setattr(i, "height", s[1]))
                self.add_widget(lbl)
        if self.time_lbl.parent is None:  # время добавляем в конец
            self.add_widget(self.time_lbl)


class Engine:
    """Потокобезопасная обёртка над pylolm.Llama."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.llm = None
        self.load_error = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.llm is not None or self.load_error is not None:
                return
            try:
                self.llm = Llama(self.model_path, n_ctx=N_CTX, verbose=False)
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"

    @property
    def ready(self):
        return self.llm is not None

    def chat_stream(self, messages):
        for chunk in self.llm.create_chat_completion(
                messages, max_tokens=MAX_TOKENS, temperature=0.7,
                stream=True):
            yield chunk

    def stop(self):
        if self.llm is not None:
            self.llm.stop()

    def reset(self):
        if self.llm is not None:
            self.llm.reset()


class ChatApp(App):
    title = "Egor AI"

    def build(self):
        self.history = []
        self.engine = Engine(MODEL_PATH)
        self.busy = False
        self.tools = False
        self._stream_bubble = None
        self._gen_started = time.time()
        Window.clearcolor = hex_color(BG)

        # --- шапка: Egor AI + статус слева, кнопки справа
        top = BoxLayout(size_hint_y=None, height=dp(58), spacing=dp(8),
                        padding=[dp(14), dp(6)])
        titles = BoxLayout(orientation="vertical", spacing=dp(2))
        name_lbl = Label(text="Egor AI", font_size=dp(18), bold=True,
                         color=hex_color(ACCENT), halign="left",
                         size_hint_y=None, height=dp(24))
        name_lbl.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
        self.status = Label(text="Загрузка модели…", font_size=dp(12),
                            color=hex_color(TEXT_SOFT), halign="left",
                            size_hint_y=None, height=dp(17))
        self.status.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
        titles.add_widget(name_lbl)
        titles.add_widget(self.status)

        btn_copy = PillButton("Копир.", bg=CARD, fg=TEXT_SOFT, font=12,
                              radius=22, size_hint_x=None, width=dp(70),
                              height=dp(44))
        btn_copy.bind(on_press=lambda _: self.copy_last())
        self.btn_tools = PillButton("Файлы: выкл", bg=CARD, fg=TEXT_SOFT,
                                    font=12, radius=22, size_hint_x=None,
                                    width=dp(92), height=dp(44))
        self.btn_tools.bind(on_press=lambda _: self.safe(self.toggle_tools))
        btn_clear = PillButton("Очистить", bg=CARD, fg=TEXT_SOFT, font=13,
                               radius=22, size_hint_x=None, width=dp(88),
                               height=dp(44))
        btn_clear.bind(on_press=lambda _: self.clear_chat())
        top.add_widget(titles)
        top.add_widget(btn_copy)
        top.add_widget(self.btn_tools)
        top.add_widget(btn_clear)

        # --- история
        self.msg_list = BoxLayout(orientation="vertical", size_hint_y=None,
                                  spacing=dp(12), padding=dp(10))
        self.msg_list.bind(minimum_height=self.msg_list.setter("height"))
        self.scroll = ScrollView(bar_width=dp(3),
                                 bar_color=hex_color("#5a544c"))
        self.scroll.add_widget(self.msg_list)

        # --- чипсы-подсказки: горизонтальная прокрутка, не вылезают
        chips_scroll = ScrollView(size_hint_y=None, height=dp(40),
                                  do_scroll_x=True, do_scroll_y=False,
                                  bar_width=0)
        chips_row = BoxLayout(size_hint=(None, None), height=dp(36),
                              spacing=dp(8), padding=[dp(10), 0])
        chips_row.bind(minimum_width=chips_row.setter("width"))
        for label in ("Расскажи о себе", "Идея для проекта", "Напиши код",
                      "Объясни просто"):
            chip = PillButton(label, bg=CARD, fg="#cfc9c0", font=12,
                              radius=19, size_hint_x=None,
                              width=dp(8 * len(label) + 26))
            chip.bind(on_press=lambda _b, t=label: self.safe(
                lambda: self.preset(t)))
            chips_row.add_widget(chip)
        chips_scroll.add_widget(chips_row)

        # --- строка ввода-пилюля
        row = BoxLayout(size_hint_y=None, height=dp(58), spacing=dp(8),
                        padding=[dp(10), dp(6)])
        self.input = TextInput(hint_text="Спросите что-нибудь…",
                               multiline=False, font_size=dp(15),
                               background_normal="", background_active="",
                               background_color=hex_color(PANEL),
                               foreground_color=hex_color(TEXT_MAIN),
                               hint_text_color=hex_color(TEXT_SOFT),
                               cursor_color=hex_color(ACCENT))
        self.input.bind(on_text_validate=lambda _: self.safe(self.send))
        self.btn_send = PillButton("Отправить", bg=ACCENT, fg="#ffffff",
                                   font=14, radius=24, size_hint_x=None,
                                   width=dp(104))
        self.btn_send.bind(on_press=lambda _: self.safe(self.send))
        self.btn_stop = PillButton("Стоп", bg="#6e3a3a", fg="#ffd9d9",
                                   font=14, radius=24, size_hint_x=None,
                                   width=dp(76), disabled=True)
        self.btn_stop.bind(on_press=lambda _: self.engine.stop())
        row.add_widget(self.input)
        row.add_widget(self.btn_send)
        row.add_widget(self.btn_stop)

        root = BoxLayout(orientation="vertical", spacing=dp(4))
        root.add_widget(top)
        root.add_widget(self.scroll)
        root.add_widget(chips_scroll)
        root.add_widget(row)

        threading.Thread(target=self._load_model, daemon=True).start()
        return root

    def safe(self, fn):
        """Выполнить действие UI; любая ошибка — в чат, а не вылет."""
        try:
            fn()
        except Exception as exc:
            try:
                self.add_system(f"Сбой интерфейса: {type(exc).__name__}: {exc}")
            except Exception:
                pass

    def _load_model(self):
        if not os.path.isfile(MODEL_PATH):
            self.engine.load_error = (f"Файл модели не найден: {MODEL_PATH}")
        else:
            self.engine.load()
        Clock.schedule_once(self._after_load)

    def _after_load(self, _dt):
        if self.engine.ready:
            self.set_status("Готов • офлайн")
            self.add_system("Привет! Я Egor AI — работаю без интернета.")
        else:
            err = self.engine.load_error or "неизвестная ошибка"
            self.set_status("Движок недоступен")
            if "WinError" in err or "LibraryNotFound" in err:
                self.add_system("Нативные библиотеки pylolm доступны только "
                                "на Android arm64 — это ок для ПК-теста UI.\n"
                                "Подробности:\n" + err)
            else:
                self.add_system("Не удалось загрузить модель:\n" + err)

    def set_status(self, text):
        self.status.text = text

    def add_system(self, text):
        self.msg_list.add_widget(Bubble("system", text))
        self._scroll_down()

    def _scroll_down(self):
        Clock.schedule_once(lambda dt: setattr(self.scroll, "scroll_y", 0),
                            0.05)

    def preset(self, text):
        """Чипс: подставить текст и отправить."""
        if self.busy or not text:
            return
        self.input.text = text
        self.send()

    def copy_last(self):
        """Плюшка: скопировать последний ответ ассистента."""
        if not self.history:
            self.add_system("Пока нечего копировать.")
            return
        for msg in reversed(self.history):
            if msg["role"] == "assistant":
                try:
                    from kivy.core.clipboard import Clipboard
                    Clipboard.copy(msg["content"])
                    self.add_system("Ответ скопирован в буфер обмена")
                except Exception as exc:
                    self.add_system(f"Не удалось скопировать: {exc}")
                return
        self.add_system("Ответов ассистента ещё нет.")

    def send(self):
        text = self.input.text.strip()
        if not text or self.busy:
            return
        if not self.engine.ready:
            self.add_system("Модель ещё не готова (" +
                            (self.engine.load_error or "загружается") + ")")
            return

        self.busy = True
        self.btn_send.disabled = True
        self.btn_stop.disabled = False
        self.set_status("Печатает…")
        self._gen_started = time.time()

        self.msg_list.add_widget(Bubble("user", text))
        self._stream_bubble = Bubble("assistant", "")
        self.msg_list.add_widget(self._stream_bubble)
        self._scroll_down()

        self.history.append({"role": "user", "content": text})
        self.input.text = ""
        threading.Thread(target=self._generate, daemon=True).start()

    def _generate(self):
        """Рабочий поток: стриминг + авто-продолжение обрывов + агентный
        цикл с файловыми командами ([CMD:...])."""
        answer, error = "", None
        steps = 0
        try:
            while steps <= MAX_AGENT_STEPS:
                steps += 1
                part, finish = "", None
                msgs = [{"role": "system",
                         "content": SYSTEM_PROMPT + (TOOLS_PROMPT
                                                     if self.tools else "")}]
                msgs += self.history
                if answer:  # продолжение после обрыва
                    msgs += [{"role": "assistant", "content": answer},
                             {"role": "user",
                              "content": "Продолжи с места обрыва."}]
                for chunk in self.engine.chat_stream(msgs):
                    ch = chunk["choices"][0]
                    piece = ch.get("delta", {}).get("content")
                    if piece:
                        part += piece
                        Clock.schedule_once(
                            lambda dt, p=piece: self._append(p))
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                answer += part

                # --- агентный шаг: есть ли команда?
                idx = answer.find("[CMD:")
                if self.tools and idx >= 0:
                    end = answer.find("[/CMD]", idx)
                    cmd_text = (answer[idx:end + 6] if end >= 0
                                else answer[idx:])
                    result = run_tool(cmd_text)
                    # модель видит полную переписку: команда + результат
                    self.history.append({"role": "assistant",
                                         "content": answer})
                    self.history.append({
                        "role": "user",
                        "content": f"РЕЗУЛЬТАТ КОМАНДЫ:\n{result}\n"
                                   "Продолжай ответ."})
                    Clock.schedule_once(
                        lambda dt, c=cmd_text, r=result:
                        self.add_system(f"КОМАНДА: {c[:60]}\n→ {r[:140]}"))
                    # видимый текст ответа — без команды
                    Clock.schedule_once(
                        lambda dt, a=answer[:idx]: self._replace_stream(a))
                    continue

                if finish != "length" or steps > MAX_CONTINUATIONS + 1:
                    break
                Clock.schedule_once(lambda dt: self._append("\n"))
        except PyLoLmError as exc:
            error = f"Ошибка генерации: {exc}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        Clock.schedule_once(lambda dt: self._finish(answer, error))

    def toggle_tools(self):
        self.tools = not self.tools
        self.btn_tools.text = "Файлы: вкл" if self.tools else "Файлы: выкл"
        self.btn_tools._bg = hex_color(ACCENT if self.tools else CARD)
        self.btn_tools._upd()
        if self.tools and not self._ensure_storage_permission():
            return  # разрешения нет — ждем, пока пользователь выдаст
        self.add_system(TOOLS_HELP if self.tools
                        else "Режим файлов выключен.")

    def _ensure_storage_permission(self):
        """Проверить доступ к общей памяти; на Android попросить его.

        MANAGE_EXTERNAL_STORAGE — не обычное разрешение, его выдают только
        через специальную страницу настроек, которую мы и открываем.
        """
        # быстрая проверка: можно ли писать в общую память
        probe = os.path.join("/sdcard", ".egorai_probe") \
            if os.path.isdir("/sdcard") else None
        if probe is None or os.access("/sdcard", os.W_OK):
            try:
                if probe:
                    open(probe, "w").close()
                    os.remove(probe)
            except OSError:
                pass
            return True
        try:
            from jnius import autoclass
            Intent = autoclass("android.content.Intent")
            Settings = autoclass("android.provider.Settings")
            Uri = autoclass("android.net.Uri")
            Activity = autoclass("org.kivy.android.PythonActivity")
            pkg = Activity.mActivity.getPackageName()
            intent = Intent(
                Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                Uri.parse("package:" + pkg))
            Activity.mActivity.startActivity(intent)
            self.add_system(
                "Android открыл страницу доступа к файлам.\n"
                "1. Включи тумблер 'Разрешить доступ ко всем файлам'\n"
                "2. Вернись в приложение\n"
                "3. Снова нажми 'Файлы: вкл'")
        except Exception as exc:
            self.add_system(
                "Не удалось открыть настройки: " + str(exc) +
                "\nВыдай доступ вручную: Настройки -> Приложения -> "
                "Egor AI -> Разрешения -> Файлы и медиа")
        return False

    def _replace_stream(self, text):
        """Заменить видимый текст текущего пузыря (после выполнения команды)."""
        if self._stream_bubble is not None:
            self._stream_bubble.label.text = text

    def _append(self, piece):
        if self._stream_bubble is not None:
            self._stream_bubble.label.text += piece

    def _finish(self, answer, error):
        self.busy = False
        self.btn_send.disabled = False
        self.btn_stop.disabled = True
        if error:
            self.set_status("Ошибка")
            self.add_system(error)
            return
        if answer:
            # убрать служебные команды из финального ответа
            idx = answer.find("[CMD:")
            clean = answer[:idx].strip() if idx >= 0 else answer
            if clean:
                self.history.append({"role": "assistant", "content": clean})
            if self._stream_bubble is not None:
                # блоки кода -> карточки с кнопкой «Копировать»
                self._stream_bubble.format_with_code(clean or answer)
            secs = max(time.time() - self._gen_started, 0.001)
            st = "Готов • офлайн"
            try:
                n = self.engine.llm.count_tokens(answer)
                st = f"Готов • {n / secs:.1f} ток/сек • ~{n} ток"
            except Exception:
                pass
            self.set_status(st)
        self._scroll_down()

    def clear_chat(self):
        if self.busy:
            return
        self.history.clear()
        self.engine.reset()
        self.msg_list.clear_widgets()
        self._stream_bubble = None
        self.add_system("Диалог очищен. Чем помочь?")

    def on_pause(self):
        return True

    def on_stop(self):
        if self.engine.llm is not None:
            self.engine.llm.close()


if __name__ == "__main__":
    ChatApp().run()
