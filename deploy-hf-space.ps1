param(
    [string]$Message = "Update Hugging Face Space"
)

$ErrorActionPreference = "Stop"

$source = "D:\FYP IMPLEMENTATION\EmpowerHer_Chatbot"
$target = "D:\FYP IMPLEMENTATION\hf-space"

Write-Host "Syncing project into hf-space..."
robocopy $source $target /E /XD venv FRONTEND\node_modules FRONTEND\dist __pycache__ .git /XF .env | Out-Null

# Robocopy exit codes 0-7 are success variants.
if ($LASTEXITCODE -gt 7) {
    throw "Robocopy failed with exit code $LASTEXITCODE"
}

Set-Location $target

Write-Host "Checking git status..."
git status --short

Write-Host "Staging changes..."
git add .

$stagedChanges = git status --porcelain
if (-not $stagedChanges) {
    Write-Host "No changes to deploy."
    exit 0
}

Write-Host "Committing changes..."
git commit -m $Message

Write-Host "Pushing to Hugging Face Space..."
git push origin main

Write-Host "Deployment push complete."
