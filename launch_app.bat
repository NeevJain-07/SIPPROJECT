@echo off
title DocDewarp - Diagnostic & Analytics Studio (Streamlit)
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo =====================================================================
echo    DocDewarp - Streamlit Diagnostic & Analytics Studio
echo =====================================================================
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Using Python: %PYTHON_EXE%
echo Starting Streamlit on http://127.0.0.1:8501 ...
echo.

"%PYTHON_EXE%" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
pause
