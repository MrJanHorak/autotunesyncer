# AutoTuneSyncer AI Coding Guidelines

## Project Overview

GPU-accelerated video composition tool for synchronizing MIDI-triggered autotune effects with video. **Hybrid Node.js/Python fullstack** app: React + Vite frontend, Express.js backend, PyTorch-based video processing.

## Critical Architecture Patterns

### Chunk-Based Processing (NOT Note-by-Note)

**NEVER** revert to note-by-note FFmpeg processing. The system uses **chunk-based** architecture after a critical migration:

- **Problem solved:** 136 MIDI notes creating 136 FFmpeg processes → timeouts, resource exhaustion
- **Current approach:** Group notes into 4-second chunks (~34 processes for 136 notes)
- **Key files:**
  - [`backend/utils/video_processor.py`](backend/utils/video_processor.py) - Drop-in replacement using `VideoComposerWrapper`
  - [`backend/python/video_composer.py`](backend/python/video_composer.py) - 6091-line core compositor with GPU acceleration
  - [`backend/python/video_composer_wrapper.py`](backend/python/video_composer_wrapper.py) - Interface adapter
- **Legacy backups:** `*_problematic_backup.py`, `*_corrupted_backup.py` - DO NOT USE

### Hybrid Backend Communication

Node.js Express server spawns Python subprocesses for video processing:

```javascript
// backend/controllers/compositionController.js
spawn('python', ['backend/python/video_composer.py', ...])
```

- Large payloads (1GB limit): [`backend/server.js`](backend/server.js) sets `express.json({ limit: '1000mb' })`
- Python receives JSON via stdin, returns progress via stdout (`PROGRESS:XX`)
- Frontend polls `/api/compose/status/:jobId` for async job updates
- **Authentication:** Currently **NO** authentication middleware. CORS is restricted to `localhost:5173`. User auth is planned for future releases.

### Async Job Queue & Caching

Heavy video processing is managed via Redis and Bull queues to prevent server blocking:

- **Queue Service** ([`backend/services/queueService.js`](backend/services/queueService.js)):
  - Uses `bull` for job management (requires Redis).
  - **Data Transformation:** Transforms job data (merging tracks/drums) into Python-compatible JSON format before spawning subprocesses.
  - **Progress Tracking:** Updates job progress which frontend polls.
- **Cache Service** ([`backend/services/cacheService.js`](backend/services/cacheService.js)):
  - Uses `ioredis` with **in-memory fallback** if Redis is unavailable.
  - Critical for storing temporary job states and results.

### Drum Processing Strategy

Drums require special handling to map single MIDI tracks to multiple video elements:

- **Identification:** Channel 10 (index 9) or instrument names containing "drum" are automatically detected via `isDrumTrack()`.
- **Individual Elements:** Unlike melodic tracks, **each drum note (Kick, Snare, Hi-Hat) is treated as a separate track**.
  - **Mapping:** `DRUM_NOTE_MAP` (in [`src/js/drumUtils.js`](src/js/drumUtils.js) and [`backend/python/drum_utils.py`](backend/python/drum_utils.py)) maps MIDI notes to specific drum names.
  - **Composition:** The backend splits a single drum MIDI track into multiple logical tracks (e.g., "Kick Track", "Snare Track") so different videos can be assigned to each drum sound.

### Audio Autotune Logic

Audio processing is handled separately from video composition but synchronized:

- **Pitch Detection:** Uses `crepe` (preferred) or `librosa` (fallback) in [`backend/python/autotune.py`](backend/python/autotune.py).
- **Pitch Shifting:** Uses `pyrubberband` (preferred) or `librosa` for high-quality time-stretching and pitch-shifting.
- **Parallel Processing:** `ParallelAutotuneProcessor` class manages thread-safe audio processing, utilizing GPU if available (via TensorFlow).

### GPU Acceleration Strategy

PyTorch + CUDA for video frame processing, FFmpeg for encoding:

- **Auto-detection:** Falls back to CPU if CUDA unavailable ([`backend/python/processing_utils.py`](backend/python/processing_utils.py))
- **Codecs:** `h264_nvenc` (NVIDIA GPU) → `libx264` (CPU fallback)
- **Conservative settings:** Removed problematic options like `hwaccel_output_format`, `tune`, `rc`, `cq` after RTX 3050 compatibility issues
- **Key imports:** `from processing_utils import GPUManager`, `from ffmpeg_gpu import ffmpeg_gpu_encode`
- **DO NOT** add advanced GPU tuning without testing on diverse hardware

### PathRegistry Singleton Pattern

Centralized file path tracking across Python modules ([`backend/python/path_registry.py`](backend/python/path_registry.py)):

```python
registry = PathRegistry.get_instance()
registry.register_instrument("piano", "C4", "path/to/video.mp4")
registry.get_instrument_path("piano", "C4")
```

- Thread-safe with `RLock`, LRU cache
- Normalizes names: `"Grand Piano"` → `"grand_piano"`
- Validates paths before registration (optional)

### Frontend MIDI Processing

React hooks manage MIDI parsing and instrument mapping ([`src/hooks/useMidiProcessing.js`](src/hooks/useMidiProcessing.js)):

- **Parsing:** `@tonejs/midi` library in [`src/components/MidiParser/MidiParser.jsx`](src/components/MidiParser/MidiParser.jsx)
- **Drums:** Channel 10 identified via `isDrumTrack()` ([`src/js/drumUtils.js`](src/js/drumUtils.js))
- **Name normalization:** Frontend `normalizeInstrumentName()` must match Python's `normalize_instrument_name()`
- **Grid layout:** Stores video arrangement in `gridArrangement` state for composition

## Development Workflow

### Running the System

```powershell
# Quick test (components + GPU check)
npm run test:quick
# OR: pwsh -ExecutionPolicy Bypass -File test-quick.ps1

# Frontend dev server (port 5173)
npm run dev

# Backend API server (port 3000)
cd backend
npm install  # Node dependencies
pip install -r requirements.txt  # Python deps
node server.js
```

### Testing Strategy

- **PowerShell test launchers:** `test-quick.ps1`, `test-dev.ps1`, `test-e2e.ps1`, `test-production.ps1`
- **Python integration tests:** `test_components.py`, `test_final_validation.py`, `test_gpu_fix.py`
- **Node tests:** `test_api_endpoint.js`, `test_complete_fixed_pipeline.js`
- **Migration validation:** `test_migration_success.py` verifies chunk-based architecture

### Key Configuration Files

- [`config.js`](config.js) - Frontend API URL, recording settings
- [`backend/python/gpu_config.py`](backend/python/gpu_config.py) - FFmpeg GPU parameters
- [`vite.config.js`](vite.config.js) - Dev server proxy to port 3000
- [`.env`](.env) - Environment variables (not in version control)
- [`backend/utils/health_monitor.py`](backend/utils/health_monitor.py) - System health monitoring script (spawned by controller, currently experimental/optional).

## Common Pitfalls

### 1. MIDI Data Format Mismatch

**VideoComposer expects `time` field, receives `start`/`end`:**

```python
# Transform before passing to compositor
note['time'] = note.get('start') or note.get('startTime')
```

### 2. Windows Path Handling

Python paths use forward slashes internally, convert at boundaries:

```python
path = Path(windows_path).as_posix()  # Convert to forward slashes
```

### 3. FFmpeg Audio Sync Issues

**Always use `-map 0:v -map 1:a` for explicit stream mapping:**

```python
# backend/python/gpu_pipeline.py
ffmpeg_cmd = [
    'ffmpeg', '-i', video_path, '-i', audio_path,
    '-map', '0:v', '-map', '1:a',  # Explicit mapping
    '-c:v', codec, '-c:a', 'aac', output_path
]
```

### 4. Large Payload Failures

Express default limits cause 413 errors. Already configured to 1GB, but verify:

```javascript
app.use(express.json({ limit: '1000mb' }));
multer({ limits: { fileSize: 1000 * 1024 * 1024 } });
```

### 5. Unicode Logging on Windows

Python logging must handle CP1252 encoding:

```python
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        if hasattr(handler.stream, 'reconfigure'):
            handler.stream.reconfigure(encoding='utf-8', errors='replace')
```

## Code Review Checklist

- [ ] Uses chunk-based processing (not note-by-note loops)
- [ ] GPU code has CPU fallback with try/except
- [ ] MIDI note data includes `time` field (not just `start`/`end`)
- [ ] FFmpeg commands use explicit `-map` for audio/video streams
- [ ] PathRegistry used for file tracking (not ad-hoc dictionaries)
- [ ] Frontend instrument names normalized to match Python backend
- [ ] Large file uploads respect 1GB payload limits
- [ ] Progress reporting uses `PROGRESS:XX` stdout format for Python → Node communication

## Useful References

- **Migration docs:** [`MIGRATION_SUCCESS_REPORT.md`](MIGRATION_SUCCESS_REPORT.md), [`MIGRATION_COMPLETE_FINAL.md`](MIGRATION_COMPLETE_FINAL.md)
- **GPU fixes:** [`GPU_FIX_SUMMARY.md`](GPU_FIX_SUMMARY.md), [`FFMPEG_IMPROVEMENTS.md`](FFMPEG_IMPROVEMENTS.md)
- **Video composition:** [`VIDEO_COMPOSITION_FIX_COMPLETE.md`](VIDEO_COMPOSITION_FIX_COMPLETE.md)
- **Main README:** [`README.md`](README.md) - Project overview and setup

## Tech Stack Quick Reference

**Frontend:** React 18.3, Vite 5.4, Tone.js (MIDI), @tonejs/midi, React Player  
**Backend Node:** Express.js, multer, fluent-ffmpeg, midi-parser-js  
**Backend Python:** PyTorch 2.1.2, MoviePy 2.0, NumPy 1.24, librosa 0.10, FFmpeg  
**GPU:** CUDA Toolkit 11.x, CuPy (optional), h264_nvenc codec
