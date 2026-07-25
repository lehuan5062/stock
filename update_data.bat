@echo off
title Vietnamese Stock Predictor - Data Fetch
cd /d "%~dp0"

echo.
echo === Vietnamese Stock Predictor - Data Fetch ===
echo Pre-fills the OHLCV cache from KBS/VCI. No model, no LLM - just data.
echo.

if not exist ".venv\Scripts\python.exe" goto :novenv

rem All prompting happens in Python (see cli.py update-data --interactive).
rem Keep this wrapper free of parenthesized blocks / delayed expansion: a
rem batch parse error aborts the script instantly, skipping the pause below,
rem which closes the window before the user can read the error.
.venv\Scripts\python.exe -m stockpredict.cli update-data --interactive

echo.
echo === Done. ===
pause
exit /b 0

:novenv
echo ERROR: virtual environment not found at .venv
echo Run setup.bat first.
pause
exit /b 1
