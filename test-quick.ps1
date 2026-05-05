# Quick Component and Performance Test
Write-Host "🔥 AutoTuneSyncer Enhanced Performance Test" -ForegroundColor Green

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title,

        [Parameter(Mandatory = $true)]
        [scriptblock]$Action
    )

    Write-Host $Title -ForegroundColor Yellow
    & $Action

    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code ${LASTEXITCODE}: $Title"
    }
}

cd "c:\Users\janny\development\autotunesyncer"

Invoke-Step "`n1️⃣  Testing Core Components..." {
    python test_components.py
}

Invoke-Step "`n2️⃣  Running Video Composer Regressions..." {
    python -m unittest -v test_video_composer_regressions
}

cd backend

Invoke-Step "`n3️⃣  Checking GPU Status..." {
    python python/gpu_setup.py
}

Invoke-Step "`n4️⃣  Verifying Python Dependencies..." {
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
}

cd ../

Invoke-Step "`n5️⃣  Checking Node.js Dependencies..." {
    node -e "console.log('Node.js version:', process.version)"
    npm list --depth=0 2>$null | Select-String "react|express|ffmpeg"
}

Write-Host "`n✅ Performance test complete!" -ForegroundColor Green
