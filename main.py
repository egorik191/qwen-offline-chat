# -*- coding: utf-8 -*-
"""
Egor AI — офлайн-чат с локальной нейросетью (Qwen2.5-3B-Instruct).

UI:      Kivy (ПК: Windows/Linux/macOS, телефоны: Android/iOS)
Движок:  pylolm — Android-совместимая обёртка над llama.cpp
Модель:  qwen2.5-3b-instruct-q4_k_m.gguf (лежит рядом с main.py)
Режим:   полностью офлайн, никаких облачных API

Архитектура:
  - поток UI (Kivy) никогда не блокируется;
  - модель грузится один раз в фоновом потоке при старте;
  - генерация идёт в отдельном потоке, токены приходят через
    Clock.schedule_once (единственный безопасный способ тронуть
    виджеты Kivy не из UI-потока);
  - кнопка «Стоп» вызывает llm.stop() — это безопасно из UI-потока;
  - если ответ оборвался по лимиту токенов, приложение само
    продолжит генерацию (до MAX_CONTINUATIONS раз) и склеит ответ.
"""

import os
import threading
import time

# ---------------------------------------------------------------------------
# ПУТИ И НАСТРОЙКА ОКРУЖЕНИЯ — до импорта pylolm!
# ---------------------------------------------------------------------------
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
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, RoundedRectangle

from pylolm import Llama, PyLoLmError  # noqa: E402  (импорт после env!)

Window.soft_input_mode = "below_target"

# ---------------------------------------------------------------- палитра
BG = "#0f1218"          # фон приложения
PANEL = "#1a1f29"       # панели и поля ввода
ACCENT = "#5b8cff"      # основной акцент
ACCENT_DIM = "#3d5fbf"
MY_BUBBLE = "#5b8cff"
AI_BUBBLE = "#222938"
SYSTEM_BUBBLE = "#2a2317"
TEXT_MAIN = "#eef1f7"
TEXT_SOFT = "#9aa3b5"

SYSTEM_PROMPT = (
    "Ты — полезный ассистент по имени Egor AI. Отвечай кратко и по-русски, "
    "если вопрос на русском. Если просят код — пиши код."
)

MAX_TOKENS = 700        # лимит токенов за один заход генерации
MAX_CONTINUATIONS = 4   # сколько раз автоматически продолжать обрыв
N_CTX = 2048


def hex_color(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [1]


# ---------------------------------------------------------------------------
# ВИДЖЕТЫ
# ---------------------------------------------------------------------------

class RoundedButton(Button):
    """Кнопка со скруглёнными углами (нативные кнопки Kivy — квадратные)."""

    def __init__(self, text, bg=ACCENT, fg="#ffffff", size=15, **kw):
        super().__init__(text=text, **kw)
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self.color = hex_color(fg)
        self.font_size = dp(size)
        self.bold = True
        self._bg = hex_color(bg)
        self._pressed = False
        with self.canvas.before:
            Color(*self._bg)
            self._rect = RoundedRectangle(pos=self.pos, size=self.size,
                                          radius=[dp(24)])
        self.bind(pos=lambda _, p: setattr(self._rect, "pos", p))
        self.bind(size=lambda _, s: setattr(self._rect, "size", s))

    def on_press(self):
        # лёгкое затемнение при нажатии
        self._rect.color = [c * 0.8 for c in self._bg[:3]] + [1]


class Bubble(BoxLayout):
    """Сообщение: скруглённый пузырь с именем автора, текстом и временем."""

    def __init__(self, role, text="", **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.size_hint = (0.86, None)
        self.padding = dp(12)
        self.spacing = dp(3)
        # высота обязана следовать за содержимым, иначе при стриминге
        # текст вылезает за фон-прямоугольник
        self.bind(minimum_height=self.setter("height"))

        name, bg, name_color = {
            "user": ("Ты", MY_BUBBLE, "#dbe6ff"),
            "assistant": ("Egor AI", AI_BUBBLE, "#8fd6a5"),
            "system": ("", SYSTEM_BUBBLE, TEXT_SOFT),
        }[role]

        if name:
            head = Label(text=name, font_size=dp(11), bold=True,
                         color=hex_color(name_color), size_hint_y=None,
                         height=dp(16), halign="left")
            head.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
            self.add_widget(head)

        self.label = Label(
            text=text, font_size=dp(15), color=hex_color(TEXT_MAIN),
            halign="left", valign="top", size_hint_y=None,
        )
        self.label.bind(
            width=lambda inst, w: setattr(inst, "text_size", (w, None)))
        self.label.bind(
            texture_size=lambda inst, s: setattr(inst, "height", s[1]))
        self.add_widget(self.label)

        if role != "system":
            self.time_lbl = Label(
                text=time.strftime("%H:%M"), font_size=dp(10),
                color=hex_color("#b9c2d4"), size_hint_y=None, height=dp(14),
                halign="right")
            self.time_lbl.bind(
                width=lambda i, w: setattr(i, "text_size", (w, None)))
            self.add_widget(self.time_lbl)

        with self.canvas.before:
            Color(*hex_color(bg))
            self._rect = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[dp(16)])
        self.bind(pos=lambda _, p: setattr(self._rect, "pos", p))
        self.bind(size=lambda _, s: setattr(self._rect, "size", s))


class ChatApp(App):
    title = "Egor AI"

    def build(self):
        self.history = []          # [{"role": ..., "content": ...}] для модели
        self.engine = Engine(MODEL_PATH)
        self.busy = False
        self._stream_bubble = None
        self._chat_started = time.time()
        self._gen_started = time.time()

        Window.clearcolor = hex_color(BG)

        # --- верхняя панель: имя, статус, «Очистить» справа
        top = BoxLayout(size_hint_y=None, height=dp(56), spacing=dp(10),
                        padding=[dp(12), dp(6)])
        titles = BoxLayout(orientation="vertical")
        name_lbl = Label(text="Egor AI", font_size=dp(17), bold=True,
                         color=hex_color(TEXT_MAIN), halign="left",
                         size_hint_y=None, height=dp(24))
        name_lbl.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
        self.status = Label(text="Загрузка модели…", font_size=dp(12),
                            color=hex_color(TEXT_SOFT), halign="left",
                            size_hint_y=None, height=dp(18))
        self.status.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
        titles.add_widget(name_lbl)
        titles.add_widget(self.status)

        btn_clear = RoundedButton("Очистить", bg=PANEL, fg=TEXT_SOFT,
                                  size=13, size_hint_x=None, width=dp(96))
        btn_clear.bind(on_press=lambda _: self.clear_chat())
        top.add_widget(titles)
        top.add_widget(btn_clear)

        # --- история сообщений
        self.msg_list = BoxLayout(orientation="vertical", size_hint_y=None,
                                  spacing=dp(10), padding=dp(10))
        self.msg_list.bind(minimum_height=self.msg_list.setter("height"))
        self.scroll = ScrollView(bar_width=dp(3),
                                 bar_color=hex_color(ACCENT_DIM))
        self.scroll.add_widget(self.msg_list)

        # --- панель быстрых подсказок («фишка»)
        chips_row = BoxLayout(size_hint_y=None, height=dp(36),
                              spacing=dp(8), padding=[dp(10), 0])
        for label in ("Расскажи о себе", "Идея для проекта", "Напиши код",
                      "Объясни просто"):
            chip = RoundedButton(label, bg=PANEL, fg=TEXT_SOFT, size=12,
                                 size_hint_x=None,
                                 width=dp(10 * len(label) + 24))
            chip.bind(on_press=lambda _, t=label: self._ask_preset(t))
            chips_row.add_widget(chip)

        # --- строка ввода
        row = BoxLayout(size_hint_y=None, height=dp(54), spacing=dp(8),
                        padding=[dp(10), dp(4)])
        self.input = TextInput(hint_text="Спросите что-нибудь…",
                               multiline=False, font_size=dp(15),
                               background_color=hex_color(PANEL),
                               foreground_color=hex_color(TEXT_MAIN),
                               hint_text_color=hex_color(TEXT_SOFT),
                               cursor_color=hex_color(ACCENT),
                               background_normal="",
                               background_active="")
        self.input.bind(on_text_validate=lambda _: self.send())
        self.btn_send = RoundedButton("➤", size=18, size_hint_x=None,
                                      width=dp(60))
        self.btn_send.bind(on_press=lambda _: self.send())
        self.btn_stop = RoundedButton("Стоп", bg="#7d2f3f", size=14,
                                      size_hint_x=None, width=dp(72),
                                      disabled=True)
        self.btn_stop.bind(on_press=lambda _: self.engine.stop())
        row.add_widget(self.input)
        row.add_widget(self.btn_send)
        row.add_widget(self.btn_stop)

        root = BoxLayout(orientation="vertical", spacing=dp(4))
        root.add_widget(top)
        root.add_widget(self.scroll)
        root.add_widget(chips_row)
        root.add_widget(row)

        # Модель грузится в фоне — UI остаётся отзывчивым
        threading.Thread(target=self._load_model, daemon=True).start()
        return root

    # ---------------------------------------------------------------- модель

    def _load_model(self):
        if not os.path.isfile(MODEL_PATH):
            self.engine.load_error = (
                f"Файл модели не найден: {MODEL_PATH}\n"
                "Проверь, что .gguf попал в сборку (source.include_exts)."
            )
        else:
            self.engine.load()
        Clock.schedule_once(self._after_load)

    def _after_load(self, _dt):
        if self.engine.ready:
            self.status.text = "Готов • офлайн"
            self.add_system("Привет! Я Egor AI — офлайн-ассистент. "
                            "Спросите что-нибудь ✨")
        else:
            err = self.engine.load_error or "неизвестная ошибка"
            self.status.text = "Движок недоступен"
            if "WinError" in err or "LibraryNotFound" in err:
                self.add_system(
                    "На этом устройстве нативные библиотеки pylolm "
                    "недоступны (они только для Android arm64).\n"
                    "На ПК проверяется интерфейс; генерация работает "
                    "на телефоне после сборки APK.\n\nПодробности:\n" + err)
            else:
                self.add_system("Не удалось загрузить модель:\n" + err)

    # ---------------------------------------------------------------- чат

    def add_system(self, text):
        self.msg_list.add_widget(Bubble("system", text))
        self._scroll_down()

    def _scroll_down(self):
        Clock.schedule_once(lambda dt: setattr(self.scroll, "scroll_y", 0),
                            0.05)

    def _ask_preset(self, text):
        """Быстрая подсказка: подставить и отправить."""
        if self.busy or not text:
            return
        self.input.text = text
        self.send()

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
        self.status.text = "Печатает…"
        self._gen_started = time.time()

        self.msg_list.add_widget(Bubble("user", text))
        self._stream_bubble = Bubble("assistant", "")
        self.msg_list.add_widget(self._stream_bubble)
        self._scroll_down()

        self.history.append({"role": "user", "content": text})
        self.input.text = ""

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
        threading.Thread(target=self._generate, args=(messages,),
                         daemon=True).start()

    def _generate(self, messages):
        """Рабочий поток: стриминг токенов + авто-продолжение обрывов.

        Если модель упёрлась в лимит токенов (finish_reason == 'length'),
        ответ обрывается на полуслове — тогда сами добавляем «продолжи»
        и генерируем дальше, склеивая текст в том же пузыре.
        """
        answer = ""
        error = None
        turns = 0
        try:
            while True:
                part = ""
                finish_reason = None
                turn_messages = (
                    [{"role": "system", "content": SYSTEM_PROMPT}]
                    + self.history
                    + ([{"role": "assistant", "content": answer}]
                       if answer else [])
                    + ([{"role": "user", "content": "Продолжи с места обрыва."}]
                       if answer else [])
                )
                for chunk in self.engine.chat_stream(turn_messages):
                    choice = chunk["choices"][0]
                    piece = choice.get("delta", {}).get("content")
                    if piece:
                        part += piece
                        # копия piece обязательна: замыкание поймало бы
                        # последнее значение
                        Clock.schedule_once(
                            lambda dt, p=piece: self._append_stream(p))
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                answer += part
                turns += 1
                if finish_reason != "length" or turns > MAX_CONTINUATIONS:
                    break
                # ответ оборван по лимиту — продолжаем
                Clock.schedule_once(
                    lambda dt: self._append_stream("\n"))
        except PyLoLmError as exc:
            error = f"Ошибка генерации: {exc}"
        except Exception as exc:  # например, нехватка RAM на слабом телефоне
            error = f"{type(exc).__name__}: {exc}"
        Clock.schedule_once(lambda dt: self._finish(answer, error))

    def _append_stream(self, piece):
        if self._stream_bubble is not None:
            lbl = self._stream_bubble.label
            lbl.text += piece  # Label сам пересчитает высоту (texture_size)

    def _finish(self, answer, error):
        self.busy = False
        self.btn_send.disabled = False
        self.btn_stop.disabled = True
        if error:
            self.status.text = "Ошибка"
            self.add_system(error)
            return
        if answer:
            self.history.append({"role": "assistant", "content": answer})
            # фишка: скорость генерации (токенов/сек) в статусе
            secs = max(time.time() - self._gen_started, 0.001)
            if self.engine.ready:
                try:
                    n_tok = self.engine.llm.count_tokens(answer)
                    self.status.text = f"Готов • {n_tok / secs:.1f} ток/сек"
                except Exception:
                    self.status.text = "Готов • офлайн"
        self._scroll_down()

    def clear_chat(self):
        if self.busy:
            return
        self.history.clear()
        self.engine.reset()  # сбросить KV-кэш, иначе модель «помнит» старое
        self.msg_list.clear_widgets()
        self._stream_bubble = None
        self.add_system("Диалог очищен. Чем помочь? ✨")

    # ---------------------------------------------------------------- Android

    def on_pause(self):
        return True  # приложение сворачивается, модель остаётся в памяти

    def on_stop(self):
        if self.engine.llm is not None:
            self.engine.llm.close()  # освободить ~2 ГБ RAM


class Engine:
    """Потокобезопасная обёртка над pylolm.Llama."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.llm = None
        self.load_error = None
        self._load_lock = threading.Lock()

    def load(self):
        """Загрузка модели. Вызывать один раз из фонового потока."""
        with self._load_lock:
            if self.llm is not None or self.load_error is not None:
                return
            try:
                self.llm = Llama(self.model_path, n_ctx=N_CTX, verbose=False)
            except Exception as exc:  # LibraryNotFound, ModelLoad, нет файла...
                self.load_error = f"{type(exc).__name__}: {exc}"

    @property
    def ready(self):
        return self.llm is not None

    def chat_stream(self, messages):
        """Генератор кусочков ответа. Вызывается в рабочем потоке."""
        for chunk in self.llm.create_chat_completion(
            messages, max_tokens=MAX_TOKENS, temperature=0.7, stream=True
        ):
            yield chunk

    def stop(self):
        if self.llm is not None:
            self.llm.stop()  # потокобезопасно, шлёт флаг в llama.cpp

    def reset(self):
        if self.llm is not None:
            self.llm.reset()  # очистить KV-кэш (забыть диалог)


if __name__ == "__main__":
    ChatApp().run()
