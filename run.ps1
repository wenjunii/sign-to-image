$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$AppScript = Join-Path $ProjectRoot "sign_to_visual.py"
$Requirements = Join-Path $ProjectRoot "requirements.txt"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating project virtual environment..."
    python -m venv (Join-Path $ProjectRoot ".venv")
}

& $VenvPython -c "import cv2, mediapipe as mp, pythonosc; assert hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing project dependencies..."
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r $Requirements
}

& $VenvPython $AppScript @args
exit $LASTEXITCODE
