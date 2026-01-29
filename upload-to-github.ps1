# GitHub Upload Script for Mental Health Prediction Project
# Author: Muhammad Farooq
# Email: mfarooqshafee333@gmail.com

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  GitHub Upload Assistant" -ForegroundColor Cyan
Write-Host "  Mental Health Prediction Project" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if Git is installed
try {
    $gitVersion = git --version
    Write-Host "✓ Git detected: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git is not installed. Please install Git first:" -ForegroundColor Red
    Write-Host "  Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Set variables
$repoUrl = "https://github.com/Muhammad-Farooq-13/mental-health-prediction.git"
$projectPath = "e:\mental healt"

Write-Host "`nProject Information:" -ForegroundColor Yellow
Write-Host "  Author: Muhammad Farooq" -ForegroundColor White
Write-Host "  Email: mfarooqshafee333@gmail.com" -ForegroundColor White
Write-Host "  GitHub: Muhammad-Farooq-13" -ForegroundColor White
Write-Host "  Repository: $repoUrl" -ForegroundColor White

# Navigate to project directory
Write-Host "`n[1/4] Navigating to project directory..." -ForegroundColor Yellow
Set-Location $projectPath
Write-Host "✓ Current directory: $(Get-Location)" -ForegroundColor Green

# Check Git status
Write-Host "`n[2/4] Checking Git status..." -ForegroundColor Yellow
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "⚠ Uncommitted changes detected" -ForegroundColor Yellow
    $response = Read-Host "Do you want to commit these changes? (y/n)"
    if ($response -eq 'y') {
        git add .
        git commit -m "Update: Additional changes before GitHub upload"
        Write-Host "✓ Changes committed" -ForegroundColor Green
    }
} else {
    Write-Host "✓ All changes committed" -ForegroundColor Green
}

# Add remote origin
Write-Host "`n[3/4] Setting up remote repository..." -ForegroundColor Yellow
$remotes = git remote -v
if ($remotes -match "origin") {
    Write-Host "⚠ Remote 'origin' already exists" -ForegroundColor Yellow
    $response = Read-Host "Do you want to update it? (y/n)"
    if ($response -eq 'y') {
        git remote remove origin
        git remote add origin $repoUrl
        Write-Host "✓ Remote updated" -ForegroundColor Green
    }
} else {
    git remote add origin $repoUrl
    Write-Host "✓ Remote added: origin -> $repoUrl" -ForegroundColor Green
}

# Rename branch to main
Write-Host "`n[4/4] Preparing to push..." -ForegroundColor Yellow
git branch -M main
Write-Host "✓ Branch renamed to 'main'" -ForegroundColor Green

# Display upload instructions
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  READY TO UPLOAD!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Before pushing, make sure you have:" -ForegroundColor Yellow
Write-Host "  1. Created the repository on GitHub" -ForegroundColor White
Write-Host "     URL: https://github.com/Muhammad-Farooq-13" -ForegroundColor White
Write-Host "  2. Named it: mental-health-prediction" -ForegroundColor White
Write-Host "  3. Did NOT initialize with README/License/.gitignore`n" -ForegroundColor White

Write-Host "To push your code to GitHub, run:" -ForegroundColor Cyan
Write-Host "  git push -u origin main`n" -ForegroundColor White

$response = Read-Host "Do you want to push now? (y/n)"
if ($response -eq 'y') {
    Write-Host "`nPushing to GitHub..." -ForegroundColor Yellow
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n========================================" -ForegroundColor Green
        Write-Host "  SUCCESS! 🎉" -ForegroundColor Green
        Write-Host "========================================`n" -ForegroundColor Green
        Write-Host "Your project is now on GitHub!" -ForegroundColor Green
        Write-Host "View it at: https://github.com/Muhammad-Farooq-13/mental-health-prediction`n" -ForegroundColor Cyan
    } else {
        Write-Host "`n✗ Push failed. Please check:" -ForegroundColor Red
        Write-Host "  - Repository exists on GitHub" -ForegroundColor Yellow
        Write-Host "  - You have push access" -ForegroundColor Yellow
        Write-Host "  - Your credentials are correct`n" -ForegroundColor Yellow
    }
} else {
    Write-Host "`nNo problem! When ready, run:" -ForegroundColor Yellow
    Write-Host "  git push -u origin main`n" -ForegroundColor White
}

Write-Host "========================================`n" -ForegroundColor Cyan
