# Quick Component and Performance Test
Write-Host "🔥 AutoTuneSyncer Enhanced Performance Test" -ForegroundColor Green

cd "c:\Users\janny\development\autotunesyncer"

Write-Host "`n1️⃣  Testing Core Components..." -ForegroundColor Yellow
python test_components.py

Write-Host "`n2️⃣  Checking GPU Status..." -ForegroundColor Yellow
cd backend
python python/gpu_setup.py

Write-Host "`n3️⃣  Verifying Python Dependencies..." -ForegroundColor Yellow
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')

try:
    import librosa
    print(f'Librosa version: {librosa.__version__}')
except:
    print('❌ Librosa not available')

try:
    import moviepy
    print(f'MoviePy version: {moviepy.__version__}')
except:
    print('❌ MoviePy not available')
"

Write-Host "`n4️⃣  Checking Node.js Dependencies..." -ForegroundColor Yellow
cd ../
node -e "console.log('Node.js version:', process.version)"
npm list --depth=0 2>$null | Select-String "react|express|ffmpeg"

Write-Host "`n✅ Performance test complete!" -ForegroundColor Green
