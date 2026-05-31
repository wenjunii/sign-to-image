param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
)

$ErrorActionPreference = "Stop"

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements.txt"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating project virtual environment..."
    python -m venv (Join-Path $ProjectRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create virtual environment."
    }
}

& $VenvPython -c "import cv2, mediapipe as mp, pythonosc, sklearn; assert hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands') and hasattr(mp.solutions, 'holistic')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing project dependencies..."
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Could not upgrade pip."
    }
    & $VenvPython -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install project dependencies."
    }

    & $VenvPython -c "import cv2, mediapipe as mp, pythonosc, sklearn; assert hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands') and hasattr(mp.solutions, 'holistic')"
    if ($LASTEXITCODE -ne 0) {
        throw "Project dependencies are installed, but runtime imports still fail."
    }
}

Write-Output $VenvPython
