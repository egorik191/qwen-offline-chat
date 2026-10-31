[app]

# Название приложения и имя пакета (package.name — без пробелов и точек)
title = Egor AI
package.name = egorai
package.domain = org.example

# Код приложения и все данные лежат в корне проекта
source.dir = .

# ЧТО ПОПАДАЁТ В APK: код + модель GGUF + нативные .so pylolm.
#   gguf — файл модели (491 МБ);
#   so   — libllama/libggml/shim из папки pylolm_libs/ (обязательно!).
# README, buildozer.spec и прочее не включаются — расширений нет в списке.
source.include_exts = py,png,jpg,kv,atlas,gguf,so,txt

version = 1.2

# Зависимости. pylolm — чистый Python (py3-none-any wheel), собирается в APK
# без рецептов; jinja2 нужен pylolm для chat-шаблонов из GGUF.
# charset_normalizer и markupsafe закреплены на pure-python версиях
# (py3-none-any): их свежие сборки публикуют android-колёса, которые
# buildozer вшивает в requirements.txt прямыми URL, а хостовой pip их
# отвергает ("not a supported wheel on this platform").
requirements = python3,kivy,pylolm,jinja2,charset_normalizer==2.1.1,markupsafe==2.0.1

orientation = portrait
fullscreen = 0

# Права на хранение. Для работы файлового агента (🛠) на Android 11+ нужен
# MANAGE_EXTERNAL_STORAGE («Все файлы») — пользователь выдаёт его в
# Настройки → Приложения → Egor AI → Разрешения.
android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,MANAGE_EXTERNAL_STORAGE

# Android arm64: единственная архитектура, для которой pylolm публикует
# нативные библиотеки. minapi 24 = минимальный Android 7.0.
android.archs = arm64-v8a
android.api = 34
android.minapi = 24
android.ndk_api = 24

# Автоматически принимать лицензии Android SDK (обязательно для CI/Actions —
# без этого сборка падает на интерактивном запросе "Accept? (y/N)")
android.accept_sdk_license = True

# Иконку buildozer подставит дефолтную; свои — положи icon.png и presplash.jpg
# в корень проекта и раскомментируй:
#icon.filename = %(source.dir)s/icon.png
#presplash.filename = %(source.dir)s/presplash.jpg

#
# --- Файлы проекта (для справки) ---
#   main.py                              — весь код приложения
#   qwen2.5-0.5b-instruct-q4_k_m.gguf    — модель Qwen (Q4_K_M)
#   pylolm_libs/*.so                     — llama.cpp для Android arm64
#

[buildozer]
log_level = 2
warn_on_root = 1
