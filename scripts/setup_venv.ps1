$ErrorActionPreference = "Stop"

Write-Host "[1/3] Creating virtual environment (.venv)..."
python -m venv .venv

Write-Host "[2/3] Upgrading pip..."
& .\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "[3/3] Installing dependencies..."
if (Test-Path requirements.txt) {
    & .\.venv\Scripts\pip.exe install -r requirements.txt
}

Write-Host "Setup complete. Run '.\.venv\Scripts\Activate.ps1' to activate."
