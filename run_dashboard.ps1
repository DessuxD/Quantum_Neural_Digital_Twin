# Quantum Neural Digital Twin - Faculty Defense Launcher
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " QUANTUM NEURAL DIGITAL TWIN - FACULTY DEFENSE DASHBOARD" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Opening Dashboard on http://localhost:5000 ..." -ForegroundColor Yellow

$env:PYTHONPATH = (Get-Location).Path
Start-Process "http://localhost:5000"
python src/web/app.py
