
# Script to sync changes to GitHub
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$commitMessage = "Auto-update: $timestamp"

Write-Host "Checking for changes..."
$status = git status --porcelain

if ($status) {
    Write-Host "Changes found. Syncing to GitHub..."
    git add .
    git commit -m "$commitMessage"
    git push origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully synced to GitHub!" -ForegroundColor Green
    } else {
        Write-Host "Error syncing to GitHub. Please check your connection and remote settings." -ForegroundColor Red
    }
} else {
    Write-Host "No changes to sync." -ForegroundColor Yellow
}

Pause
