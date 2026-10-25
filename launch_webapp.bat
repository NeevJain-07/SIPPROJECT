@echo off
title Document Dewarping & Scanner Web Studio
cd /d "%~dp0"

echo =====================================================================
echo    Document Dewarping & Scanner Web Studio (Flask)
echo =====================================================================
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Using Python: %PYTHON_EXE%
echo Starting server on http://127.0.0.1:5000 ...
echo Opening your web browser...
echo.

:: Automatically open browser after launching
start "" http://127.0.0.1:5000

"%PYTHON_EXE%" server.py

if errorlevel 1 (
    echo.
    echo [NOTE] If dependencies are missing, please run setup.bat first.
    echo.
)
pause
