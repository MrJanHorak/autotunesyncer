# End-to-End Testing Script
Write-Host "🎯 AutoTuneSyncer End-to-End Test" -ForegroundColor Green

cd "c:\Users\janny\development\autotunesyncer"

# Step 1: Component Health Check
Write-Host "`n1️⃣  Component Health Check..." -ForegroundColor Yellow
python test_components.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Component test failed! Fix issues before proceeding." -ForegroundColor Red
    exit 1
}

# Step 2: Start Backend
Write-Host "`n2️⃣  Starting Backend Server..." -ForegroundColor Yellow
cd backend
$backendJob = Start-Job -ScriptBlock {
    cd "c:\Users\janny\development\autotunesyncer\backend"
    npm start
}

Start-Sleep -Seconds 5

# Step 3: Test Backend Endpoints
Write-Host "`n3️⃣  Testing Backend Endpoints..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:3000/api/midi" -Method GET -TimeoutSec 10
    Write-Host "✅ Backend API responding" -ForegroundColor Green
} catch {
    Write-Host "❌ Backend API not responding: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 4: Test Video Processing
Write-Host "`n4️⃣  Testing Video Processing Pipeline..." -ForegroundColor Yellow
cd ../
python -c "
import sys
sys.path.append('backend')
try:
    from python.video_composer import VideoComposer
    from python.autotune import AutotuneProcessor
    print('✅ Video processing modules loaded successfully')
    
    # Test GPU acceleration
    import torch
    if torch.cuda.is_available():
        print('✅ GPU acceleration ready')
    else:
        print('⚠️  CPU mode (GPU not available)')
        
except Exception as e:
    print(f'❌ Video processing test failed: {e}')
"

# Step 5: Performance Monitoring Test
Write-Host "`n5️⃣  Testing Performance Monitoring..." -ForegroundColor Yellow
python -c "
import sys
sys.path.append('backend/utils')
try:
    from health_monitor import HealthMonitor
    monitor = HealthMonitor()
    monitor.start_monitoring()
    import time
    time.sleep(2)
    stats = monitor.get_current_stats()
    monitor.stop_monitoring()
    print(f'✅ Performance monitoring working')
    print(f'   CPU: {stats.get(\"cpu_percent\", \"N/A\")}%')
    print(f'   Memory: {stats.get(\"memory_percent\", \"N/A\")}%')
    if 'gpu_percent' in stats:
        print(f'   GPU: {stats[\"gpu_percent\"]}%')
except Exception as e:
    print(f'❌ Performance monitoring failed: {e}')
"

Write-Host "`n6️⃣  Cleanup..." -ForegroundColor Yellow
Stop-Job $backendJob -ErrorAction SilentlyContinue
Remove-Job $backendJob -ErrorAction SilentlyContinue

Write-Host "`n🎉 End-to-end test complete!" -ForegroundColor Green
Write-Host "Ready for full application testing!" -ForegroundColor Cyan
