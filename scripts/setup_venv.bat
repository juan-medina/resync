@echo off
setlocal

echo [1/3] Creating virtual environment (.venv)...
python -m venv .venv
if %ERRORLEVEL% neq 0 (
    echo Failed to create virtual environment. Ensure Python is installed and added to PATH.
    exit /b %ERRORLEVEL%
)

echo [2/3] Upgrading pip...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo [3/3] Installing dependencies...
if exist requirements.txt (
    pip install -r requirements.txt
)

echo Setup complete. Run ".venv\Scripts\activate.bat" to activate.
