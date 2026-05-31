param(
    [string]$Data = "data/clips",
    [string]$Output = "models/temporal_sign_model.pkl",
    [int]$Frames = 48,
    [double]$Seconds = 1.4,
    [double]$MaxMissingSeconds = 0.18,
    [int]$Estimators = 400,
    [double]$TestSize = 0.25,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = & (Join-Path $ProjectRoot "scripts\ensure_venv.ps1") -ProjectRoot $ProjectRoot | Select-Object -Last 1

& $VenvPython (Join-Path $ProjectRoot "train_temporal_model.py") `
    --data $Data `
    --output $Output `
    --frames $Frames `
    --seconds $Seconds `
    --max-missing-seconds $MaxMissingSeconds `
    --estimators $Estimators `
    --test-size $TestSize `
    --seed $Seed
exit $LASTEXITCODE
