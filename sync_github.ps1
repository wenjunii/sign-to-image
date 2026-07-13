param(
    [string]$RemoteUrl = "https://github.com/wenjunii/sign-to-image.git",
    [string]$Branch = "main",
    [switch]$Push
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Push-Location $ProjectRoot
try {
    if (-not (Test-Path ".git")) {
        throw "This directory is not a Git repository: $ProjectRoot"
    }

    git remote get-url origin *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Adding origin remote: $RemoteUrl"
        git remote add origin $RemoteUrl
    } else {
        $CurrentRemote = git remote get-url origin
        if ($CurrentRemote -ne $RemoteUrl) {
            Write-Host "Updating origin remote:"
            Write-Host "  from $CurrentRemote"
            Write-Host "  to   $RemoteUrl"
            git remote set-url origin $RemoteUrl
        }
    }

    Write-Host "Fetching origin..."
    git fetch origin

    Write-Host ""
    git remote -v
    git status --short --branch

    if ($Push) {
        Write-Host ""
        Write-Host "Pushing $Branch to origin..."
        git push origin $Branch
    } else {
        Write-Host ""
        Write-Host "Remote check complete. Use '.\sync_github.ps1 -Push' to push $Branch."
    }
} finally {
    Pop-Location
}
