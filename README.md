# Qwen 2.5 Офлайн-Чат (Kivy + pylolm, APK через Buildozer)

Локальный ИИ-чат без облака: Qwen2.5-0.5B-Instruct (Q4_K_M, GGUF) крутится
прямо на телефоне через llama.cpp (обёртка **pylolm**). UI — Kivy.

```
qwen-chat/
├── main.py                              # всё приложение (чат, потоки, стриминг)
├── buildozer.spec                       # конфиг сборки APK
├── qwen2.5-0.5b-instruct-q4_k_m.gguf    # модель (491 МБ, уже в проекте)
└── pylolm_libs/                         # .so llama.cpp для Android arm64 (11 .so)
```

## Как это работает

- **Потоки:** модель грузится и генерирует в фоновом `threading.Thread`,
  UI-поток Kivy не блокируется. Токены приходят в интерфейс только через
  `Clock.schedule_once` — единственный потокобезопасный способ.
- **Путь к модели:** `os.path.join(os.path.dirname(os.path.abspath(__file__)), "…gguf")`.
  На ПК это папка с `main.py`; в APK buildozer кладёт файлы из `source.dir`
  в приложение, python-for-android распаковывает их при первом запуске в
  `/data/data/<приложение>/files/`, и `__file__` уже указывает на реальный
  путь. Один и тот же код работает и там, и там.
- **Нативные библиотеки:** p4a не умеет ставить Android-колёса, поэтому
  `.so` из релиза pylolm лежат в `pylolm_libs/` проекта, попадают в APK
  через `source.include_exts = …,so`, а `main.py` до импорта pylolm ставит
  `PYLOLM_LIB_DIR` на эту папку (официально рекомендованный автором pylolm
  способ для Kivy).

## Тест на ПК (Windows)

```bash
pip install kivy pylolm
python main.py
```

Окно откроется, интерфейс полностью рабочий, но в чате появится сообщение,
что нативные библиотеки доступны только для Android arm64 — это нормально:
**pylolm публикует .so только под Android arm64** (и Termux), Windows-сборок
нет. Полный тест с генерацией — на телефоне.

## Сборка APK (Linux / WSL2)

На Windows buildozer не работает — нужен Linux (WSL2, любой Linux-ПК или
бесплатный Google Colab). Папку `qwen-chat/` целиком скопировать туда
(модель и .so уже внутри).

```bash
sudo apt update && sudo apt install -y python3-pip git zip unzip openjdk-17-jdk autoconf libtool pkg-config cmake
pip install buildozer cython
cd qwen-chat
buildozer -v android debug        # первая сборка: 30–60 мин, скачает SDK/NDK (~3 ГБ)
```

Готовый APK: `bin/qwenchat-1.0-arm64-v8a-debug.apk` (~500 МБ из-за модели).

Установка на телефон: перекинуть APK по USB/мессенджеру → открыть на
телефоне → разрешить «установку из неизвестных источников». Либо по USB:

```bash
adb install bin/qwenchat-1.0-arm64-v8a-debug.apk
```

## Где взять модель (если нужна заново)

Файл `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (491 МБ) — модель уже лежит в
проекте. Оригинал: репозиторий `Qwen/Qwen2.5-0.5B-Instruct-GGUF` на Hugging
Face (скачивание файла модели с HF — это не облачный API, а просто файл).

## Возможные проблемы

| Симптом | Причина и решение |
|---|---|
| Первая сборка > 30 мин | Норма: качаются Android SDK/NDK (~3 ГБ) и компилируются зависимости. Дальше — быстро (кэш в `.buildozer/`). |
| `LibraryNotFoundError` в приложении | `.so` не попали в APK: проверь `source.include_exts = …,so` и папку `pylolm_libs/` рядом с `main.py`. |
| `ModelLoadError` на телефоне | Не хватает RAM (нужно ~600 МБ свободных): закрой приложения, перезапусти. Если не помогает — уменьши `N_CTX` в `main.py` до 1024. |
| APK ~500 МБ | Внутри модель 491 МБ — так и задумано. Альтернатива: убрать `gguf` из include_exts, скачать модель на телефон в `/sdcard/` и указать этот путь в `MODEL_PATH` (потребуются разрешения на хранение, они уже в spec). |
| Ответ очень медленный | 0.5B на CPU телефона даёт ~3–10 ток/с — это ожидаемо. Ускорит: `n_threads` = число ядер (pylolm ставит до 4 сам). |
| `pylolm`/`jinja2` не ставятся в buildozer | В requirements именно `python3,kivy,pylolm,jinja2` (pylolm — pure-python wheel, рецептов p4a не требует; jinja2 тянет markupsafe — соберётся из sdist). |
| Ошибка «Failed building wheel for …» | Обнови cython (`pip install -U cython`) и buildozer; сотри `.buildozer/` и собери заново. |
| Приложение сразу закрывается | Смотри лог: `adb logcat | grep -i python`. Чаще всего — модель не попала в APK (см. выше). |
