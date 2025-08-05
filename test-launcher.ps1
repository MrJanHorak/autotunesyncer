# AutoTuneSyncer Test Launcher
Write-Host "🎵 AutoTuneSyncer Enhanced Testing Suite" -ForegroundColor Green
Write-Host "Choose your testing approach:" -ForegroundColor White

Write-Host "`n1️⃣   Quick Component Test (2 minutes)" -ForegroundColor Yellow
Write-Host "     Tests all enhanced components and GPU acceleration"

Write-Host "`n2️⃣   Development Environment (Full Setup)" -ForegroundColor Yellow  
Write-Host "     Starts frontend + backend for interactive testing"

Write-Host "`n3️⃣   End-to-End Test (5 minutes)" -ForegroundColor Yellow
Write-Host "     Complete pipeline test with performance monitoring"

Write-Host "`n4️⃣   Production Build Test (10 minutes)" -ForegroundColor Yellow
Write-Host "     Build and test production-ready version"

Write-Host "`n5️⃣   Components Only (30 seconds)" -ForegroundColor Yellow
Write-Host "     Just run the component integration test"

$choice = Read-Host "`nEnter your choice (1-5)"

switch ($choice) {
    "1" { 
        Write-Host "`n🔥 Running Quick Test..." -ForegroundColor Green
        .\test-quick.ps1 
    }
    "2" { 
        Write-Host "`n🚀 Starting Development Environment..." -ForegroundColor Green
        .\test-dev.ps1 
    }
    "3" { 
        Write-Host "`n🎯 Running End-to-End Test..." -ForegroundColor Green
        .\test-e2e.ps1 
    }
    "4" { 
        Write-Host "`n🏭 Running Production Test..." -ForegroundColor Green
        .\test-production.ps1 
    }
    "5" { 
        Write-Host "`n🧪 Running Component Test..." -ForegroundColor Green
        python test_components.py 
    }
    default { 
        Write-Host "❌ Invalid choice. Please run the script again." -ForegroundColor Red 
    }
}
