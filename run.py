#!/usr/bin/env python3
"""Launcher: تشغيل ai-service (FastAPI) محلياً عبر "python run.py".

ينشئ بيئة افتراضية (.venv) إن لم تكن موجودة، يثبّت requirements.txt،
ثم يشغّل تطبيق FastAPI عبر uvicorn (app.main:app) باستخدام إعدادات .env.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
IS_WINDOWS = sys.platform == "win32"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def run(args: list[str]) -> None:
    result = subprocess.run(args, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    if not (ROOT / ".env").exists():
        print('⚠️  لم يتم العثور على ملف ".env" في ai-service/ — انسخه من .env.example وعدّل المفاتيح قبل الاستمرار.')

    if not VENV_PYTHON.exists():
        print("📦 إنشاء بيئة افتراضية (.venv)...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])

    print("📦 تثبيت حزم ai-service (pip install -r requirements.txt)...")
    run([str(VENV_PYTHON), "-m", "pip", "install", "-q", "-r", "requirements.txt"])

    print("🚀 تشغيل ai-service (FastAPI/uvicorn)...")
    run([str(VENV_PYTHON), "-m", "app.main"])


if __name__ == "__main__":
    main()
