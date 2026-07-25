@echo off
setlocal enabledelayedexpansion
title Vietnamese Stock Predictor - Setup
cd /d "%~dp0"

echo.
echo === Vietnamese Stock Predictor - Setup ===
echo Creates/updates .venv and installs all dependencies.
echo.

where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: the "py" launcher was not found. Install Python 3.10-3.14 from python.org
  echo make sure "Install launcher for all users" is checked, then re-run this script.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" goto :create_venv

echo An existing .venv was found.
set /p REINSTALL=Reinstall/upgrade dependencies in place? y/n [y]:
if "!REINSTALL!"=="" set REINSTALL=y
if /I not "!REINSTALL!"=="y" (
  echo Setup skipped. .venv left as-is.
  pause
  exit /b 0
)
goto :install_deps

:create_venv
echo Creating virtual environment at .venv ...
py -3.13 -m venv .venv
if errorlevel 1 (
  echo ERROR: failed to create .venv with Python 3.13.
  echo Install Python 3.13 from python.org and re-run this script.
  pause
  exit /b 1
)

:install_deps
echo.
echo Upgrading pip ...
.venv\Scripts\python.exe -m pip install -U pip
if errorlevel 1 (
  echo ERROR: pip upgrade failed.
  pause
  exit /b 1
)

echo.
echo Installing stockpredict + dependencies (dev, llm extras) ...
.venv\Scripts\python.exe -m pip install -e ".[dev,llm]"
if errorlevel 1 (
  echo ERROR: dependency installation failed. See output above.
  pause
  exit /b 1
)

echo.
echo === Installed versions ===
.venv\Scripts\python.exe -m pip show vnstock vnai pandas numpy pyarrow pyyaml click matplotlib tqdm anthropic pytest pytest-cov 2>nul | findstr /R "^Name ^Version"

echo.
if not exist ".env" if exist ".env.example" echo Optional: copy .env.example to .env and fill in API keys for LLM modes.

echo.
echo === Setup complete. Paste agent_prompt.md into your LLM agent to start, ===
echo === or run update_data.bat first to pre-fill the cache. ===
pause
