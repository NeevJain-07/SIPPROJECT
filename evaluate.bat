@echo off
title DocDewarp - Benchmark Evaluator
cd /d "%~dp0"

echo =====================================================================
echo    Running Document Dewarping Quantitative Benchmark (evaluate.py)
echo =====================================================================
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Using Python: %PYTHON_EXE%
echo.

"%PYTHON_EXE%" evaluate.py
echo.
pause
