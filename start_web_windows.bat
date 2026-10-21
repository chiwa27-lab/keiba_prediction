@echo off
rem Windows: double-click to start the prediction web page.
cd /d "%~dp0"

if not exist ".venv\setup_done.txt" (
  echo First run: installing required software. This takes 3-10 minutes...
  if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if not exist ".venv\Scripts\python.exe" python -m venv .venv
  )
  if not exist ".venv\Scripts\python.exe" goto nopython
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -e ".[web]"
  if errorlevel 1 goto failed
  echo done> ".venv\setup_done.txt"
)

if not exist "data\nankan.sqlite" (
  echo data\nankan.sqlite was not found. Put the "data" folder next to this file.
  pause
  exit /b 1
)

echo.
echo Starting. The browser will open in a few seconds.
echo To quit, close this window.
echo.
".venv\Scripts\python.exe" -m nankan_keiba web
pause
exit /b 0

:nopython
echo Python 3.12 was not found. Install it from https://www.python.org/downloads/windows/
echo (check "Add python.exe to PATH" in the installer), then run this file again.
pause
exit /b 1

:failed
echo Installation failed. Check your internet connection and run this file again.
pause
exit /b 1
