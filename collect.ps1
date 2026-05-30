param(
    [Parameter(Mandatory=$true)]
    [string]$Label,
    [int]$Count = 30,
    [double]$Seconds = 1.4,
    [int]$Camera = 0,
    [string]$Output = "data/clips",
    [switch]$NoMirror
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = & (Join-Path $ProjectRoot "scripts\ensure_venv.ps1") -ProjectRoot $ProjectRoot | Select-Object -Last 1
$ArgsList = @(
    (Join-Path $ProjectRoot "collect_gesture_clips.py"),
    "--label", $Label,
    "--count", $Count,
    "--seconds", $Seconds,
    "--camera", $Camera,
    "--output", $Output
)

if ($NoMirror) {
    $ArgsList += "--no-mirror"
}

& $VenvPython @ArgsList
exit $LASTEXITCODE
