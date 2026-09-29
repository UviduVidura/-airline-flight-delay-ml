$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python 3.11 virtual environment at .venv..."
    py -3.11 -m venv .venv
}

if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating the virtual environment..."
    . .\.venv\Scripts\Activate.ps1
}

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "Setup complete. Activate the environment with: .\.venv\Scripts\Activate.ps1"
