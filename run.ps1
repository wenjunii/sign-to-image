$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = & (Join-Path $ProjectRoot "scripts\ensure_venv.ps1") -ProjectRoot $ProjectRoot | Select-Object -Last 1

& $VenvPython (Join-Path $ProjectRoot "sign_to_visual.py") @args
exit $LASTEXITCODE
