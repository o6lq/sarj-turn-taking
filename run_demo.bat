@echo off
REM Windows: double-click to start the demo. It opens http://127.0.0.1:8790/demo by itself.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo No .venv found. Run setup_windows.ps1 first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m tools.live
pause
