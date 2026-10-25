@echo off
title Document Dewarping & Scanner - Environment Setup
cd /d "%~dp0"

echo =====================================================================
echo    Document Dewarping & Scan Quality Assessment System
echo    One-Click Environment Setup & Dependency Installation
echo =====================================================================
echo.

:: 1. Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.9, 3.10, 3.11, or 3.12 from https://www.python.org/
    echo Be sure to check the box "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/3] Detected system Python:
python --version
echo.

:: 2. Create virtual environment if it does not exist
if not exist .venv (
    echo [2/3] Creating virtual environment (.venv)...
    python -m venv .venv
    if errorlevel 1 (
        echo [WARNING] Failed to create .venv. Will install packages to active Python environment.
        set "PYTHON_EXE=python"
    ) else (
        echo Virtual environment created successfully.
        set "PYTHON_EXE=.venv\Scripts\python.exe"
    )
) else (
    echo [2/3] Virtual environment (.venv) already exists.
    set "PYTHON_EXE=.venv\Scripts\python.exe"
)
echo.

:: 3. Upgrade pip and install requirements
echo [3/3] Installing / verifying dependencies from requirements.txt...
"%PYTHON_EXE%" -m pip install --upgrade pip
"%PYTHON_EXE%" -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [WARNING] Some dependencies could not be installed automatically.
    echo Please check your internet connection or run:
    echo     pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo    Setup Completed Successfully!
echo =====================================================================
echo.
echo You can now launch either interface:
echo   - run_app.bat      : Launches the Interactive Web Studio (Flask)
echo   - launch_app.bat   : Launches the Diagnostic Analytics Studio (Streamlit)
echo   - evaluate.bat     : Runs the Quantitative Benchmark Evaluator
echo.
pause
