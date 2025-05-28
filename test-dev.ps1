# AutoTuneSyncer Development Testing Script
Write-Host "🎵 Starting AutoTuneSyncer Development Environment..." -ForegroundColor Green

# Function to check if a process is running on a port
function Test-Port {
    param([int]$Port)
    try {
        $connection = Test-NetConnection -ComputerName localhost -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue
        return $connection
    }
    catch {
        return $false
    }
}

# Start Backend Server (Port 3000)
Write-Host "🔧 Starting Backend Server..." -ForegroundColor Yellow
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd 'c:\Users\janny\development\autotunesyncer\backend'; npm start"

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Start Frontend Development Server (Port 5173)
Write-Host "🌐 Starting Frontend Development Server..." -ForegroundColor Yellow
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd 'c:\Users\janny\development\autotunesyncer'; npm run dev"

# Wait for servers to start
Write-Host "⏳ Waiting for servers to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

# Check if servers are running
Write-Host "🔍 Checking server status..." -ForegroundColor Cyan

if (Test-Port 3000) {
    Write-Host "✅ Backend server running on http://localhost:3000" -ForegroundColor Green
} else {
    Write-Host "❌ Backend server not responding" -ForegroundColor Red
}

if (Test-Port 5173) {
    Write-Host "✅ Frontend server running on http://localhost:5173" -ForegroundColor Green
} else {
    Write-Host "❌ Frontend server not responding" -ForegroundColor Red
}

# Run component test
Write-Host "🧪 Running component integration test..." -ForegroundColor Magenta
cd "c:\Users\janny\development\autotunesyncer"
python test_components.py

Write-Host "`n🎉 Development environment ready!" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "Backend:  http://localhost:3000" -ForegroundColor White
Write-Host "`nPress any key to open the application..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Open the application
Start-Process "http://localhost:5173"
