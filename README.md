# AutoTuneSyncer

**A GPU-accelerated video composition tool for synchronizing MIDI-triggered autotune effects with video backgrounds.**

## Project Overview
AutoTuneSyncer is a fullstack web application that enables musicians and video creators to generate music videos by synchronizing MIDI note events with autotuned audio and dynamic video backgrounds. The system processes MIDI files, applies pitch correction, and composites timestamped video segments based on note triggers—all accelerated with GPU computing for real-time performance.

## Core Functionality
1. **MIDI Note Analysis**: Parse MIDI files to extract note events, velocities, and timing
2. **Autotune Audio Processing**: Apply pitch correction to audio based on MIDI note data
3. **Note-Triggered Video Composition**: Map MIDI notes to video clips and compose synchronized output
4. **GPU-Accelerated Rendering**: Use CUDA/PyTorch for parallel video frame processing
5. **Web-Based Interface**: React frontend with drag-and-drop MIDI/video upload and real-time preview

## Architecture

### Frontend (React + Vite)
- **Framework**: React 18.3 with Vite for fast HMR (Hot Module Replacement)
- **UI Components**: Drag-and-drop interface for MIDI and video file uploads
- **MIDI Visualization**: Real-time MIDI playback and note event display using Tone.js and @tonejs/midi
- **Video Preview**: React Player for video playback and composition preview
- **Entry Point**: [`index.html`](index.html) → React app bootstrapped in [`src/`](src/)

### Backend (Node.js + Python Hybrid)
- **Node.js Server** ([`backend/server.js`](backend/server.js)): Express.js API handling HTTP requests, CORS, and routing
- **Python Video Processing** ([`backend/python/video_composer.py`](backend/python/video_composer.py)): GPU-accelerated video composition using PyTorch and FFmpeg
- **API Routes**:
  - `/api/midi` - MIDI file parsing and note extraction
  - `/api/video` - Video upload and metadata extraction
  - `/api/compose` - Video composition orchestration (calls Python subprocess)
  - `/api/autotune` - Autotune processing (pitch correction)
  - `/api/upload` - File upload handling with large payload support

### Technologies Stack

**Frontend:**
- React 18.3, Vite 5.4
- Tone.js (MIDI playback), @tonejs/midi (MIDI parsing)
- Axios (HTTP client), React Player (video playback)
- @dnd-kit (drag-and-drop), lucide-react (icons)

**Backend:**
- **Node.js**: Express.js, CORS, midi-parser-js
- **Python 3.8+**: PyTorch (GPU tensors), FFmpeg (video encoding), NumPy
- **GPU Acceleration**: CUDA Toolkit 11.x, CuPy (optional for array operations)
- **Audio Processing**: aubio (pitch detection), Tone.js (autotune synthesis)

## Project Structure

```
├── index.html                          # Entry point for React app
├── package.json                        # Frontend dependencies (React, Vite, Tone.js)
├── vite.config.js                      # Vite build configuration
├── config.js                           # Application configuration (API URLs, paths)
│
├── src/                                # React frontend source
│   ├── components/                     # UI components (MIDI upload, video grid, etc.)
│   ├── hooks/                          # Custom React hooks
│   └── App.jsx                         # Main React application
│
├── backend/
│   ├── server.js                       # Express.js API server (port 3000)
│   ├── package.json                    # Backend Node.js dependencies
│   ├── requirements.txt                # Python dependencies (PyTorch, numpy, etc.)
│   │
│   ├── routes/                         # Express route handlers
│   │   ├── midiRoutes.js               # MIDI parsing endpoints
│   │   ├── videoRoutes.js              # Video upload/metadata endpoints
│   │   ├── composition.js              # Video composition orchestration
│   │   ├── autotuneRoutes.js           # Autotune processing endpoints
│   │   └── uploadRoutes.js             # File upload handling
│   │
│   ├── python/
│   │   └── video_composer.py           # GPU-accelerated video compositor (6100 lines)
│   │
│   ├── services/                       # Business logic layer
│   ├── controllers/                    # Request handlers
│   ├── middleware/                     # Express middleware (auth, validation)
│   └── utils/                          # Shared utilities
│
├── public/                             # Static assets
└── test_*.py, test_*.js                # Integration and unit tests
```

## Key Components & Data Flow

### 1. MIDI Note Extraction
**Files**: [`backend/routes/midiRoutes.js`](backend/routes/midiRoutes.js)
- User uploads MIDI file via frontend
- `midi-parser-js` parses binary MIDI data
- Extract note events: `{pitch, velocity, startTime, duration}`
- Return JSON array of note events to frontend

### 2. Autotune Processing
**Files**: [`backend/routes/autotuneRoutes.js`](backend/routes/autotuneRoutes.js), Tone.js in frontend
- User uploads audio file
- MIDI note data determines target pitches
- Apply pitch correction using Tone.js or Python audio libraries
- Return autotuned audio file

### 3. Video Composition (Core Pipeline)
**Files**: [`backend/python/video_composer.py`](backend/python/video_composer.py), [`backend/routes/composition.js`](backend/routes/composition.js)

**Flow**:
1. Frontend sends composition request with:
   - MIDI note events
   - Video background clips
   - Note-to-video mapping rules
2. Express calls Python subprocess: `python backend/python/video_composer.py <config_json>`
3. Python loads video clips into PyTorch GPU tensors
4. For each MIDI note event:
   - Fetch corresponding video clip
   - Crop/resize frame using GPU operations
   - Composite onto background at note timestamp
5. Encode final video with FFmpeg
6. Return video file path to frontend

### 4. GPU Acceleration
**Files**: [`backend/python/video_composer.py`](backend/python/video_composer.py), [`backend/verify_cuda.py`](backend/verify_cuda.py)
- Frame tensors stored on GPU: `torch.cuda.FloatTensor`
- Parallel batch processing: 100+ frames simultaneously
- Fallback to CPU if GPU unavailable
- See [`GPU_FIX_SUMMARY.md`](GPU_FIX_SUMMARY.md) for optimization history

## Installation

### Prerequisites

**Required:**
- **Python 3.8+** (for video processing backend)
- **Node.js 16+** (for Express API and Vite frontend)
- **FFmpeg** (for video encoding/decoding)

**Optional (Highly Recommended):**
- **CUDA Toolkit 11.x** (for GPU acceleration - 10-50x speedup)
- **NVIDIA GPU** with compute capability 3.5+

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-dev python3-pip nodejs npm
```

**Windows:**
- Install [Python](https://www.python.org/downloads/) (3.8+)
- Install [Node.js](https://nodejs.org/) (16+)
- Install [FFmpeg](https://ffmpeg.org/download.html) and add to PATH
- Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- (Optional) Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Project Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/autotunesyncer.git
cd autotunesyncer
```

2. **Install Frontend Dependencies:**
```bash
npm install
```

3. **Install Backend Node.js Dependencies:**
```bash
cd backend
npm install
cd ..
```

4. **Install Python Dependencies:**
```bash
cd backend
pip install -r requirements.txt
cd ..
```

5. **Verify GPU Setup (Optional but Recommended):**
```bash
cd backend
python verify_cuda.py
# Should print: "CUDA Available: True" and GPU details
cd ..
```

### Environment Configuration

**Backend Memory Settings:**
For processing large videos (>100MB), configure Node.js memory limits:

```bash
# Linux/macOS
export NODE_OPTIONS="--max-old-space-size=8192"

# Windows PowerShell
$env:NODE_OPTIONS="--max-old-space-size=8192"
```

**Optional Environment Variables:**
Create a `.env` file in the project root:
```bash
# Video processing
VIDEO_TEMP_DIR=./backend/temp
VIDEO_OUTPUT_DIR=./backend/processed_videos
MAX_VIDEO_SIZE_MB=500

# GPU settings
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# API configuration
BACKEND_PORT=3000
FRONTEND_PORT=5173
```

## Running the Application

### Development Mode (Recommended)

**Terminal 1 - Backend API Server:**
```bash
cd backend
npm start
# Express server runs on http://localhost:3000
```

**Terminal 2 - Frontend Dev Server:**
```bash
npm run dev
# Vite dev server runs on http://localhost:5173
# Open browser to http://localhost:5173
```

### Production Mode

**Build Frontend:**
```bash
npm run build
# Generates optimized build in dist/
```

**Serve Production Build:**
```bash
npm run preview
```

## API Endpoints

Base URL: `http://localhost:3000/api`

### MIDI Endpoints

**POST `/api/midi/parse`**
Parse MIDI file and extract note events.

**Request:**
```javascript
// FormData with MIDI file
const formData = new FormData();
formData.append('midiFile', file);
```

**Response:**
```json
{
  "notes": [
    {"pitch": 60, "velocity": 80, "startTime": 0.0, "duration": 0.5},
    {"pitch": 64, "velocity": 75, "startTime": 0.5, "duration": 0.5}
  ],
  "tempo": 120,
  "duration": 30.5
}
```

### Video Endpoints

**POST `/api/video/upload`**
Upload video file and extract metadata.

**Response:**
```json
{
  "filename": "background.mp4",
  "duration": 15.3,
  "width": 1920,
  "height": 1080,
  "fps": 30
}
```

### Composition Endpoints

**POST `/api/compose/generate`**
Generate composed video from MIDI and video clips.

**Request:**
```json
{
  "midiData": { /* parsed MIDI notes */ },
  "videoClips": [
    {"noteRange": [60, 64], "clipPath": "background1.mp4"},
    {"noteRange": [65, 72], "clipPath": "background2.mp4"}
  ],
  "outputFilename": "final_composition.mp4"
}
```

**Response:**
```json
{
  "status": "success",
  "outputPath": "backend/processed_videos/final_composition.mp4",
  "duration": 45.2,
  "processTime": 12.5
}
```

## Testing

### Quick Validation
```bash
npm run test:quick
# Runs: test-quick.ps1 - Basic API connectivity tests
```

### Component Tests
```bash
npm run test:components
# Runs: test_components.py - Tests Python video compositor
```

### End-to-End Tests
```bash
npm run test:e2e
# Runs: test-e2e.ps1 - Full pipeline validation
```

### Manual Testing Files
- **[`test_note_triggered_integration.py`](test_note_triggered_integration.py)**: Tests MIDI note-to-video mapping
- **[`test_video_composition.py`](test_video_composition.py)**: Tests video compositor output quality
- **[`test_gpu_fix.py`](test_gpu_fix.py)**: Validates GPU acceleration
- **[`test_complete_pipeline_final.py`](test_complete_pipeline_final.py)**: End-to-end integration test

## Common Development Tasks

### Adding a New MIDI Processing Feature
1. **Backend**: Add route handler in [`backend/routes/midiRoutes.js`](backend/routes/midiRoutes.js)
2. **Frontend**: Create React component in [`src/components/`](src/components/)
3. **Integration**: Update API calls in frontend using Axios
4. **Test**: Add test case in `test_*.py` or `test_*.js` files

### Modifying Video Composition Logic
1. **Core Logic**: Edit [`backend/python/video_composer.py`](backend/python/video_composer.py)
   - Frame processing: Line ~500-1500 (GPU tensor operations)
   - Video encoding: Line ~2000-2500 (FFmpeg integration)
2. **API Interface**: Update [`backend/routes/composition.js`](backend/routes/composition.js) if API contract changes
3. **Test**: Run [`test_video_composition.py`](test_video_composition.py) to verify output

### Optimizing GPU Performance
1. **Check GPU Utilization**: Run [`backend/verify_cuda.py`](backend/verify_cuda.py)
2. **Profile Code**: See [`backend/python/video_composer.py`](backend/python/video_composer.py) - cProfile integration enabled
3. **Batch Size Tuning**: Adjust `BATCH_SIZE` constant in video_composer.py
4. **Memory Management**: See [`GPU_FIX_SUMMARY.md`](GPU_FIX_SUMMARY.md) for common issues

### Adding a New Video Effect
1. Create effect function in [`backend/python/video_composer.py`](backend/python/video_composer.py):
```python
def apply_custom_effect(frame_tensor):
    # frame_tensor: torch.cuda.FloatTensor [H, W, 3]
    # Apply GPU-accelerated transformation
    return transformed_tensor
```
2. Register effect in composition pipeline (line ~3000)
3. Add frontend UI control in [`src/components/`](src/components/)

## Troubleshooting

### GPU Not Detected
**Symptoms**: Slow video processing (>1 minute for 30-second video)

**Solutions**:
```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Run full diagnostic
cd backend
python verify_cuda.py
```

**Expected Output**:
```
CUDA Available: True
GPU Device: NVIDIA GeForce RTX 3080
GPU Memory: 10GB
```

### FFmpeg Encoding Errors
**Symptoms**: Video composition fails with "FFmpeg error" or "Codec not found"

**Solutions**:
```bash
# Verify FFmpeg installation
ffmpeg -version

# Test video encoding
ffmpeg -i input.mp4 -c:v libx264 -preset fast output.mp4

# Windows: Ensure FFmpeg in PATH
# Add C:\ffmpeg\bin to System Environment Variables
```

### Memory Errors (OOM - Out of Memory)
**Symptoms**: "RuntimeError: CUDA out of memory" or Node.js heap limit errors

**Solutions**:

**For GPU Memory**:
```bash
# Reduce batch size in video_composer.py
# Edit line ~50: BATCH_SIZE = 50  # Reduce from 100

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

**For Node.js Heap**:
```bash
# Increase Node.js memory (before starting server)
node --max-old-space-size=8192 backend/server.js
```

### CORS Errors in Browser Console
**Symptoms**: "Access-Control-Allow-Origin" errors when making API calls

**Solutions**:
1. Check [`backend/server.js`](backend/server.js) CORS configuration (line ~15-25)
2. Verify frontend URL matches CORS origin: `http://localhost:5173`
3. If using different port, update `origin` in CORS config

### MIDI File Not Parsing
**Symptoms**: "Invalid MIDI file" or empty note array

**Solutions**:
1. Verify MIDI file format (should be Standard MIDI File, not proprietary format)
2. Test with simple MIDI file first
3. Check [`backend/routes/midiRoutes.js`](backend/routes/midiRoutes.js) error logs
4. Use MIDI validator: `python -c "import midi; midi.read_midifile('test.mid')"`

## Migration & Fix History

This project has undergone several major refactoring and optimization phases:

- **[`MIGRATION_COMPLETE_FINAL.md`](MIGRATION_COMPLETE_FINAL.md)**: Chunk-based video processing migration
- **[`GPU_FIX_SUMMARY.md`](GPU_FIX_SUMMARY.md)**: GPU acceleration optimizations and bug fixes
- **[`VIDEO_COMPOSITION_FIX_COMPLETE.md`](VIDEO_COMPOSITION_FIX_COMPLETE.md)**: Video compositor stability improvements
- **[`FIXES_COMPLETE_SUMMARY.md`](FIXES_COMPLETE_SUMMARY.md)**: Comprehensive bug fix changelog
- **[`NOTE_TRIGGERED_INTEGRATION_COMPLETE.md`](NOTE_TRIGGERED_INTEGRATION_COMPLETE.md)**: MIDI note-to-video mapping implementation

These documents provide context for architectural decisions and can help AI tools understand the evolution of the codebase.

## Performance Benchmarks

**Hardware**: NVIDIA RTX 3080 (10GB), Intel i7-11700K, 32GB RAM

| Task | Input | GPU Time | CPU Time | Speedup |
|------|-------|----------|----------|---------|
| 30s video composition | 3 MIDI notes, 1080p backgrounds | 2.5s | 45s | 18x |
| 60s video composition | 10 MIDI notes, 4K backgrounds | 8.2s | 180s | 22x |
| MIDI parsing | 500-note file | 0.1s | 0.1s | 1x (I/O bound) |
| Autotune processing | 30s audio | 1.2s | 3.5s | 2.9x |

**Note**: Performance scales linearly with GPU VRAM. Larger videos may require batch size adjustment.

## Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

**Key areas for contribution:**
- Additional video effects and filters
- Alternative autotune algorithms
- UI/UX improvements
- Performance optimizations
- Cross-platform compatibility

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgements

- **MIDI Parser JS**: [github.com/colxi/midi-parser-js](https://github.com/colxi/midi-parser-js)
- **Tone.js**: Audio synthesis and MIDI playback
- **PyTorch**: GPU-accelerated tensor operations
- **FFmpeg**: Video encoding/decoding
- **React & Vite**: Modern frontend development

## Reference Links

- [MIDI File Format Specifications](https://github.com/colxi/midi-parser-js/wiki/MIDI-File-Format-Specifications)
- [PyTorch CUDA Programming](https://pytorch.org/docs/stable/notes/cuda.html)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Tone.js API](https://tonejs.github.io/)

---

## AI Tool Integration Guide

**This section helps AI assistants understand the project context:**

### Key Concepts
- **MIDI Notes → Video Mapping**: Each MIDI note event triggers a video clip at a specific timestamp
- **GPU Acceleration**: Critical for real-time performance; all frame operations use PyTorch CUDA tensors
- **Hybrid Architecture**: Node.js handles HTTP/routing, Python handles heavy GPU computation
- **Large File Handling**: Videos can be 100MB+, requiring special memory management

### Common Modification Patterns

**When asked to modify video processing:**
- **File**: [`backend/python/video_composer.py`](backend/python/video_composer.py)
- **Pattern**: Always use PyTorch GPU tensors (`torch.cuda.FloatTensor`)
- **Memory**: Call `torch.cuda.empty_cache()` after large operations
- **Testing**: Run [`test_video_composition.py`](test_video_composition.py)

**When asked to modify MIDI handling:**
- **File**: [`backend/routes/midiRoutes.js`](backend/routes/midiRoutes.js)
- **Pattern**: Use `midi-parser-js` library for parsing
- **Data Format**: Return `{notes: [{pitch, velocity, startTime, duration}]}`
- **Testing**: Test with sample MIDI files in `temp_test_data/`

**When asked to modify API:**
- **File**: [`backend/server.js`](backend/server.js) + route files
- **Pattern**: Always set high payload limits (`limit: '1000mb'`)
- **CORS**: Update allowed origins if frontend URL changes
- **Testing**: Use [`test_api_endpoint.js`](test_api_endpoint.js)

**When asked to modify UI:**
- **Files**: [`src/`](src/) React components
- **Pattern**: Use Axios for API calls, handle large file uploads
- **State**: Use React hooks for component state
- **Testing**: Manual testing via `npm run dev`

### Codebase Navigation Shortcuts

- **Video encoding logic**: [`backend/python/video_composer.py`](backend/python/video_composer.py) lines 2000-2500
- **GPU frame processing**: [`backend/python/video_composer.py`](backend/python/video_composer.py) lines 500-1500
- **MIDI note extraction**: [`backend/routes/midiRoutes.js`](backend/routes/midiRoutes.js) lines 10-100
- **API routing**: [`backend/server.js`](backend/server.js) lines 30-45
- **Frontend entry**: [`src/App.jsx`](src/App.jsx) (main React component)

