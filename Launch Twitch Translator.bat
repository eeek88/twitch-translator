@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py
if %errorlevel% neq 0 (
    echo.
    echo The app exited with an error - see above.
    pause
)
