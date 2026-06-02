param(
    [string]$ModelUrl = "https://huggingface.co/sign/kaggle-asl-signs-1st-place/resolve/main/model.tflite",
    [string]$LabelMapUrl = "https://huggingface.co/sign/kaggle-asl-signs-1st-place/resolve/main/sign_to_prediction_index_map.json",
    [string]$ModelPath = "models/gislr_model.tflite",
    [string]$LabelMapPath = "models/sign_to_prediction_index_map.json",
    [int]$TargetFrames = 64,
    [double]$WindowSeconds = 1.6,
    [int]$Threads = 1,
    [switch]$Force,
    [switch]$SkipEnvUpdate
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = & (Join-Path $ProjectRoot "scripts\ensure_venv.ps1") -ProjectRoot $ProjectRoot | Select-Object -Last 1

function Resolve-ProjectPath {
    param([string]$InputPath)
    if ([System.IO.Path]::IsPathRooted($InputPath)) {
        return $InputPath
    }
    return Join-Path $ProjectRoot $InputPath
}

function Install-LiteRt {
    & $VenvPython -c "import ai_edge_litert.interpreter" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "LiteRT runtime already installed."
        return
    }

    Write-Host "Installing LiteRT runtime..."
    & $VenvPython -m pip install ai-edge-litert
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install ai-edge-litert."
    }
}

function Download-FileIfNeeded {
    param(
        [string]$Url,
        [string]$OutFile,
        [string]$Name
    )

    $Parent = Split-Path -Parent $OutFile
    if ($Parent -and -not (Test-Path $Parent)) {
        New-Item -ItemType Directory -Force -Path $Parent | Out-Null
    }

    if ((Test-Path $OutFile) -and -not $Force) {
        Write-Host "$Name already exists: $OutFile"
        return
    }

    Write-Host "Downloading $Name..."
    Invoke-WebRequest -Uri $Url -OutFile $OutFile
}

function Set-EnvValue {
    param(
        [string]$EnvPath,
        [string]$Name,
        [string]$Value
    )

    if (-not (Test-Path $EnvPath)) {
        Copy-Item (Join-Path $ProjectRoot ".env.example") $EnvPath
    }

    $Lines = @(Get-Content $EnvPath)
    $Found = $false
    $EscapedName = [regex]::Escape($Name)
    for ($Index = 0; $Index -lt $Lines.Count; $Index++) {
        if ($Lines[$Index] -match "^\s*$EscapedName\s*=") {
            $Lines[$Index] = "$Name=$Value"
            $Found = $true
        }
    }
    if (-not $Found) {
        $Lines += "$Name=$Value"
    }
    Set-Content -Path $EnvPath -Value $Lines -Encoding UTF8
}

function Normalize-EnvPath {
    param([string]$InputPath)
    if ([System.IO.Path]::IsPathRooted($InputPath)) {
        return $InputPath
    }
    return ($InputPath -replace "\\", "/")
}

$ResolvedModelPath = Resolve-ProjectPath $ModelPath
$ResolvedLabelMapPath = Resolve-ProjectPath $LabelMapPath

Push-Location $ProjectRoot
try {
    Install-LiteRt
    Download-FileIfNeeded -Url $ModelUrl -OutFile $ResolvedModelPath -Name "GISLR/PopSign TFLite model"
    Download-FileIfNeeded -Url $LabelMapUrl -OutFile $ResolvedLabelMapPath -Name "GISLR/PopSign label map"

    Write-Host "Validating GISLR/PopSign model..."
    & $VenvPython -c "import sys, numpy as np; from gislr_tflite import GislrTfliteModel, load_gislr_label_map, GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS; model=GislrTfliteModel(sys.argv[1], num_threads=int(sys.argv[3])); labels=load_gislr_label_map(sys.argv[2]); frames=int(sys.argv[4]); scores=model.predict(np.zeros((frames, GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS), dtype=np.float32)); print(f'runtime={model.runtime} threads={model.num_threads} labels={len(labels)} outputs={scores.shape[0]}')" $ResolvedModelPath $ResolvedLabelMapPath $Threads $TargetFrames
    if ($LASTEXITCODE -ne 0) {
        throw "GISLR/PopSign model validation failed."
    }

    & $VenvPython -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "Python dependency check failed."
    }

    if (-not $SkipEnvUpdate) {
        $EnvPath = Join-Path $ProjectRoot ".env"
        Set-EnvValue -EnvPath $EnvPath -Name "SIGN_RECOGNITION_BACKEND" -Value "gislr_tflite"
        Set-EnvValue -EnvPath $EnvPath -Name "SIGN_LANDMARK_PIPELINE" -Value "holistic"
        Set-EnvValue -EnvPath $EnvPath -Name "SIGN_MODEL_PATH" -Value (Normalize-EnvPath $ModelPath)
        Set-EnvValue -EnvPath $EnvPath -Name "SIGN_GISLR_LABEL_MAP" -Value (Normalize-EnvPath $LabelMapPath)
        Set-EnvValue -EnvPath $EnvPath -Name "SIGN_GISLR_TARGET_FRAMES" -Value ([string]$TargetFrames)
        Set-EnvValue -EnvPath $EnvPath -Name "SIGN_GISLR_WINDOW_SECONDS" -Value ([string]$WindowSeconds)
        Set-EnvValue -EnvPath $EnvPath -Name "SIGN_GISLR_THREADS" -Value ([string]$Threads)
        Set-EnvValue -EnvPath $EnvPath -Name "SIGN_COMMIT_MODE" -Value "manual"
        Write-Host "Updated local .env for GISLR/PopSign manual testing."
    }

    Write-Host ""
    Write-Host "GISLR/PopSign setup complete."
    Write-Host "Run: .\run.ps1"
} finally {
    Pop-Location
}
