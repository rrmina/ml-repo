@echo off
REM ML Project Setup Script using UV (Windows Batch)
REM For Windows users who prefer CMD over PowerShell

setlocal enabledelayedexpansion

REM Get project name from user or use default
set PROJECT_NAME=%1
if "%PROJECT_NAME%"=="" (
    set /p PROJECT_NAME="Enter project name (default: my_ml_project): "
    if "!PROJECT_NAME!"=="" set PROJECT_NAME=my_ml_project
)

echo [INFO] Setting up ML project: %PROJECT_NAME%

REM Check if UV is installed
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] UV is not installed. Please install UV manually:
    echo.
    echo PowerShell:
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo Or visit: https://github.com/astral-sh/uv
    echo.
    pause
    exit /b 1
) else (
    echo [SUCCESS] UV is already installed
)

REM Initialize UV project
echo [INFO] Initializing UV project...
uv init %PROJECT_NAME%
cd %PROJECT_NAME%

REM Set Python version
echo [INFO] Setting Python version to 3.11...
echo 3.11> .python-version

REM Create project structure
echo [INFO] Creating project structure...

REM Create main Python files
type nul > train.py
type nul > eval.py
type nul > main.py
type nul > inference.py

REM Create module directories
if not exist "data" mkdir data
if not exist "models" mkdir models
type nul > data\__init__.py
type nul > models\__init__.py
if not exist "checkpoints" mkdir checkpoints

REM Remove default hello.py if it exists
if exist hello.py del hello.py

echo [SUCCESS] Project structure created

echo [INFO] For full setup, please run the PowerShell script instead:
echo   powershell -ExecutionPolicy Bypass -File ..\setup.ps1 %PROJECT_NAME%
echo.
echo Or manually complete the setup:
echo   1. uv add torch numpy pandas scikit-learn matplotlib tqdm
echo   2. uv add --dev pytest black ruff jupyter ipython
echo   3. uv sync
echo.

pause
