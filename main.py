# -*- coding: utf-8 -*-
"""
Чат с локальным ИИ — Qwen2.5-0.5B-Instruct прямо на устройстве.

UI:      Kivy (ПК: Windows/Linux/macOS, телефоны: Android/iOS)
Движок:  pylolm — Android-совместимая обёртка над llama.cpp
Модель:  qwen2.5-0.5b-instruct-q4_k_m.gguf (лежит рядом с main.py)
Режим:   полностью офлайн, никаких облачных API

Архитектура:
  - поток UI (Kivy) никогда не блокируется;
  - модель грузится один раз в фоновом потоке при старте;
  - генерация идёт в отдельном потоке, токены приходят через
    Clock.schedule_once (единственный безопасный способ тронуть
    виджеты Kivy не из UI-потока);
  - кнопка «Стоп» вызывает llm.stop() — это безопасно из UI-потока.
"""

import os
import threading

# ---------------------------------------------------------------------------
# ПУТИ И НАСТРОЙКА ОКРУЖЕНИЯ — до импорта pylolm!
# ---------------------------------------------------------------------------
# BASE_DIR работает одинаково:
#   на ПК            -> папка, где лежит main.py;
#   внутри APK       -> Buildozer копирует файлы из source.dir в APK, а при
#                       первом запуске python-for-android распаковывает их в
#                       /data/data/<приложение>/files/, и __file__ указывает
#                       уже на РЕАЛЬНЫЙ путь в файловой системе. Поэтому
#                       модель и .so-библиотеки находятся тем же кодом, что и
#                       на компьютере, — никакой special-case логики.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf")

# pylolm: где искать нативные библиотеки llama.cpp (.so). В APK они лежат
# рядом с main.py (попадают туда через source.include_exts = ...,so).
os.environ.setdefault("PYLOLM_LIB_DIR", os.path.join(BASE_DIR, "pylolm_libs"))
# Запрещаем pylolm что-либо скачивать в рантайме: в APK всё уже внутри.
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

from pylolm import Llama, PyLoLmError  # noqa: E402  (импорт после env!)

Window.soft_input_mode = "below_target"  # клавиатура не закрывает ввод

ACCENT = "#3f6cff"
MY_BUBBLE = "#3f6cff"
AI_BUBBLE = "#252b36"
SYSTEM_BUBBLE = "#3a2f22"

SYSTEM_PROMPT = (
    "Ты — полезный ассистент. Отвечай кратко и по-русски, если вопрос на русском."
)
MAX_TOKENS = 300          # длина ответа (токенов)
N_CTX = 2048              # размер контекста модели


def hex_color(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [1]


# ---------------------------------------------------------------------------
# ДВИЖОК (обёртка над pylolm; живёт в фоновом потоке)
# ---------------------------------------------------------------------------

class Engine:
    """Потокобезопасная обёртка над pylolm.Llama.

    pylolm сам сериализует вызовы локом, поэтому один экземпляр можно
    использовать и для генерации, и для stop() из UI-потока.
    """

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
            piece = chunk["choices"][0]["delta"].get("content")
            if piece:
                yield piece

    def stop(self):
        if self.llm is not None:
            self.llm.stop()  # потокобезопасно, шлёт флаг в llama.cpp

    def reset(self):
        if self.llm is not None:
            self.llm.reset()  # очистить KV-кэш (забыть диалог)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class Bubble(BoxLayout):
    """Одно сообщение: пузырь с текстом. Возвращает свой Label для стриминга."""

    def __init__(self, role, text="", **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.size_hint = (0.85, None)
        self.padding = dp(10)
        self.spacing = dp(2)

        self.label = Label(
            text=text, font_size=dp(15), color="#ffffff",
            halign="left", valign="top", size_hint_y=None,
        )
        self.label.bind(
            width=lambda inst, w: setattr(inst, "text_size", (w, None)))
        self.label.bind(
            texture_size=lambda inst, s: setattr(inst, "height", s[1]))
        self.add_widget(self.label)

        bg = {"user": MY_BUBBLE, "assistant": AI_BUBBLE,
              "system": SYSTEM_BUBBLE}.get(role, SYSTEM_BUBBLE)
        from kivy.graphics import Color, RoundedRectangle
        with self.canvas.before:
            Color(*hex_color(bg))
            self._rect = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[dp(14)])
        self.bind(pos=lambda _, p: setattr(self._rect, "pos", p))
        self.bind(size=lambda _, s: setattr(self._rect, "size", s))


class ChatApp(App):
    title = "Qwen 2.5 — офлайн-чат"

    def build(self):
        self.history = []          # [{"role": ..., "content": ...}] для модели
        self.engine = Engine(MODEL_PATH)
        self.busy = False
        self._stream_bubble = None  # пузырь текущего ответа

        # --- строка статуса и кнопки
        top = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        self.status = Label(text="Загрузка модели…", font_size=dp(13),
                            color="#9aa3b5", halign="left")
        self.status.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
        btn_clear = Button(text="🗑", size_hint_x=None, width=dp(48),
                           background_normal="", background_color=hex_color("#2b3242"))
        btn_clear.bind(on_press=lambda _: self.clear_chat())
        top.add_widget(self.status)
        top.add_widget(btn_clear)

        # --- история сообщений
        self.msg_list = BoxLayout(orientation="vertical", size_hint_y=None,
                                  spacing=dp(6), padding=dp(8))
        self.msg_list.bind(minimum_height=self.msg_list.setter("height"))
        scroll = ScrollView()
        scroll.add_widget(self.msg_list)

        # --- строка ввода
        row = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(8))
        self.input = TextInput(hint_text="Спросите что-нибудь…",
                               multiline=False, font_size=dp(15),
                               background_color=hex_color("#1d222b"),
                               foreground_color="#ffffff")
        self.input.bind(on_text_validate=lambda _: self.send())
        self.btn_send = Button(text="Отправить", size_hint_x=None, width=dp(96),
                               background_normal="",
                               background_color=hex_color(ACCENT), bold=True)
        self.btn_send.bind(on_press=lambda _: self.send())
        self.btn_stop = Button(text="Стоп", size_hint_x=None, width=dp(72),
                               background_normal="",
                               background_color=hex_color("#8a2f3f"),
                               disabled=True)
        self.btn_stop.bind(on_press=lambda _: self.engine.stop())
        row.add_widget(self.input)
        row.add_widget(self.btn_send)
        row.add_widget(self.btn_stop)

        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))
        root.add_widget(top)
        root.add_widget(scroll)
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
            self.status.text = (
                f"Готово — Qwen2.5-0.5B • контекст {N_CTX} • офлайн")
            self.add_system("Модель загружена. Спросите что-нибудь!")
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
        bubble = Bubble("system", text)
        self.msg_list.add_widget(bubble)
        self._scroll_down()

    def _scroll_down(self):
        Clock.schedule_once(lambda dt: setattr(
            self.root.children[1], "scroll_y", 0), 0.05)

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
        self.status.text = "Генерация…"

        # пузырь пользователя и пустой пузырь ответа
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
        """Рабочий поток: стриминг токенов. UI трогаем только через Clock."""
        answer = ""
        error = None
        try:
            for piece in self.engine.chat_stream(messages):
                answer += piece
                # копия piece обязательна: замыкание поймало бы последнее значение
                Clock.schedule_once(
                    lambda dt, p=piece: self._append_stream(p))
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
        self.status.text = "Готово — Qwen2.5-0.5B • офлайн"
        self._scroll_down()

    def clear_chat(self):
        if self.busy:
            return
        self.history.clear()
        self.engine.reset()  # сбросить KV-кэш, иначе модель «помнит» старое
        self.msg_list.clear_widgets()
        self._stream_bubble = None
        self.add_system("Диалог очищен.")

    # ---------------------------------------------------------------- Android

    def on_pause(self):
        return True  # приложение сворачивается, модель остаётся в памяти

    def on_stop(self):
        if self.engine.llm is not None:
            self.engine.llm.close()  # освободить ~500 МБ RAM


if __name__ == "__main__":
    ChatApp().run()
