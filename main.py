# -*- coding: utf-8 -*-
"""
Egor AI — офлайн-чат с локальной нейросетью (Qwen2.5-3B-Instruct).
Kivy + pylolm (llama.cpp). Полностью офлайн.
"""

import os
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
MAX_TOKENS = 700
MAX_CONTINUATIONS = 4
N_CTX = 2048


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
            "assistant": ("✳ Egor AI", AI_BUBBLE, "#d9a583"),
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
            t = Label(text=time.strftime("%H:%M"), font_size=dp(10),
                      color=hex_color("#8f897f"), size_hint_y=None,
                      height=dp(13), halign="right")
            t.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
            self.add_widget(t)

        with self.canvas.before:
            Color(*hex_color(bg))
            self._rect = RoundedRectangle(pos=self.pos, size=self.size,
                                          radius=[dp(18)])
        self.bind(pos=lambda _, p: setattr(self._rect, "pos", p))
        self.bind(size=lambda _, s: setattr(self._rect, "size", s))


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
        self._stream_bubble = None
        self._gen_started = time.time()
        Window.clearcolor = hex_color(BG)

        # --- шапка: ✳ Egor AI + статус слева, кнопки справа
        top = BoxLayout(size_hint_y=None, height=dp(58), spacing=dp(8),
                        padding=[dp(14), dp(6)])
        titles = BoxLayout(orientation="vertical", spacing=dp(2))
        name_lbl = Label(text="✳ Egor AI", font_size=dp(18), bold=True,
                         color=hex_color(ACCENT), halign="left",
                         size_hint_y=None, height=dp(24))
        name_lbl.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
        self.status = Label(text="Загрузка модели…", font_size=dp(12),
                            color=hex_color(TEXT_SOFT), halign="left",
                            size_hint_y=None, height=dp(17))
        self.status.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
        titles.add_widget(name_lbl)
        titles.add_widget(self.status)

        btn_copy = PillButton("⧉", bg=CARD, fg=TEXT_SOFT, font=16, radius=22,
                              size_hint_x=None, width=dp(44),
                              height=dp(44))
        btn_copy.bind(on_press=lambda _: self.copy_last())
        btn_clear = PillButton("Очистить", bg=CARD, fg=TEXT_SOFT, font=13,
                               radius=22, size_hint_x=None, width=dp(96),
                               height=dp(44))
        btn_clear.bind(on_press=lambda _: self.clear_chat())
        top.add_widget(titles)
        top.add_widget(btn_copy)
        top.add_widget(btn_clear)

        # --- история
        self.msg_list = BoxLayout(orientation="vertical", size_hint_y=None,
                                  spacing=dp(12), padding=dp(10))
        self.msg_list.bind(minimum_height=self.msg_list.setter("height"))
        self.scroll = ScrollView(bar_width=dp(3),
                                 bar_color=hex_color("#5a544c"))
        self.scroll.add_widget(self.msg_list)

        # --- чипсы-подсказки
        chips_row = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(8),
                              padding=[dp(10), 0])
        for label in ("Расскажи о себе", "Идея для проекта", "Напиши код",
                      "Объясни просто"):
            chip = PillButton(label, bg=CARD, fg="#cfc9c0", font=12,
                              radius=19, size_hint_x=None,
                              width=dp(8 * len(label) + 26))
            # ВАЖНО: обработчик только через bind (без on_press-override),
            # и через безопасную обёртку — исключение попадёт в чат, не в крах
            chip.bind(on_press=lambda _b, t=label: self.safe(
                lambda: self.preset(t)))
            chips_row.add_widget(chip)

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
        root.add_widget(chips_row)
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
            self.add_system("Привет! Я Egor AI — работаю без интернета. "
                            "Спросите что-нибудь ✨")
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
                    self.add_system("Ответ скопирован в буфер обмена ⧉")
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
        """Рабочий поток: стриминг + авто-продолжение при обрыве по лимиту."""
        answer, error, turns = "", None, 0
        try:
            while True:
                part, finish = "", None
                msgs = ([{"role": "system", "content": SYSTEM_PROMPT}]
                        + self.history)
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
                turns += 1
                if finish != "length" or turns > MAX_CONTINUATIONS:
                    break
                Clock.schedule_once(lambda dt: self._append("\n"))
        except PyLoLmError as exc:
            error = f"Ошибка генерации: {exc}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        Clock.schedule_once(lambda dt: self._finish(answer, error))

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
            self.history.append({"role": "assistant", "content": answer})
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
        self.add_system("Диалог очищен. Чем помочь? ✨")

    def on_pause(self):
        return True

    def on_stop(self):
        if self.engine.llm is not None:
            self.engine.llm.close()


if __name__ == "__main__":
    ChatApp().run()
