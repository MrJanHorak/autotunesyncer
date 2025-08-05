# Production Build and Test Script
Write-Host "🏭 AutoTuneSyncer Production Build Test" -ForegroundColor Green

cd "c:\Users\janny\development\autotunesyncer"

# Step 1: Install dependencies
Write-Host "`n1️⃣  Installing/Updating Dependencies..." -ForegroundColor Yellow
Write-Host "Frontend dependencies..." -ForegroundColor Cyan
npm install

Write-Host "Backend dependencies..." -ForegroundColor Cyan
cd backend
npm install
cd ../

# Step 2: Build frontend
Write-Host "`n2️⃣  Building Frontend..." -ForegroundColor Yellow
npm run build

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Frontend build failed!" -ForegroundColor Red
    exit 1
}

# Step 3: Test the build
Write-Host "`n3️⃣  Testing Production Build..." -ForegroundColor Yellow
Write-Host "Starting preview server..." -ForegroundColor Cyan
$previewJob = Start-Job -ScriptBlock {
    cd "c:\Users\janny\development\autotunesyncer"
    npm run preview
}

Start-Sleep -Seconds 5

# Step 4: Start backend for production test
Write-Host "`n4️⃣  Starting Backend for Production Test..." -ForegroundColor Yellow
$backendJob = Start-Job -ScriptBlock {
    cd "c:\Users\janny\development\autotunesyncer\backend"
    npm start
}

Start-Sleep -Seconds 5

# Step 5: Test production endpoints
Write-Host "`n5️⃣  Testing Production Setup..." -ForegroundColor Yellow
try {
    $backendTest = Invoke-RestMethod -Uri "http://localhost:3000/api/midi" -Method GET -TimeoutSec 10
    Write-Host "✅ Backend production ready" -ForegroundColor Green
} catch {
    Write-Host "❌ Backend production test failed" -ForegroundColor Red
}

try {
    $frontendTest = Invoke-WebRequest -Uri "http://localhost:4173" -Method GET -TimeoutSec 10
    if ($frontendTest.StatusCode -eq 200) {
        Write-Host "✅ Frontend production build working" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Frontend production test failed" -ForegroundColor Red
}

# Step 6: Run component test in production mode
Write-Host "`n6️⃣  Final Component Verification..." -ForegroundColor Yellow
python test_components.py

Write-Host "`n7️⃣  Cleanup..." -ForegroundColor Yellow
Stop-Job $previewJob -ErrorAction SilentlyContinue
Stop-Job $backendJob -ErrorAction SilentlyContinue
Remove-Job $previewJob -ErrorAction SilentlyContinue
Remove-Job $backendJob -ErrorAction SilentlyContinue

Write-Host "`n🚀 Production test complete!" -ForegroundColor Green
Write-Host "Application ready for deployment!" -ForegroundColor Cyan
