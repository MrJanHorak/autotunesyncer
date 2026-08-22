# Symphovie

**A GPU-accelerated video composition tool for synchronizing MIDI-triggered autotune effects with video backgrounds.**

> **Status note:** This project is feature-complete for personal/free use. Monetization is on hold pending copyright/ToS review — see [Legal](#legal) below.

> **Naming note:** The product is branded **Symphovie** (a blend of "symphony" and "movie") across the live UI. The repo, npm package name, and most backend module paths still use the original working name **autotunesyncer** — that's fine as an internal/technical identifier as long as it stays consistent; it doesn't need to match the product name. This README uses "Symphovie" for the product and "AutoTuneSyncer"/`autotunesyncer` only when referring to the repo or package itself.

## Project Overview

Symphovie is a fullstack web application that enables musicians and video creators to generate music videos by synchronizing MIDI note events with autotuned audio and dynamic video backgrounds. The system processes MIDI files, applies pitch correction, and composites timestamped video segments based on note triggers — accelerated with GPU computing where available.

## Core Functionality

1. **MIDI Note Analysis**: Parse MIDI files to extract note events, velocities, and timing
2. **Autotune Audio Processing**: Apply pitch correction to audio based on MIDI note data (CREPE if installed, librosa fallback otherwise)
3. **Note-Triggered Video Composition**: Map MIDI notes to video clips and compose synchronized output via a background job queue
4. **GPU-Accelerated Rendering**: Use CUDA/PyTorch and NVENC where a GPU is available; falls back to CPU (`libx264`) otherwise
5. **Accounts, Projects & Collaboration**: JWT-authenticated accounts, saved projects, and project invite links for collaborators
6. **Web-Based Interface**: React frontend with drag-and-drop MIDI/video upload and real-time preview

## Architecture

### Frontend (React + Vite)

- **Framework**: React 18.3 with Vite 5.4 for fast HMR
- **UI Components**: Drag-and-drop interface for MIDI and video file uploads
- **MIDI Visualization**: Real-time MIDI playback and note event display using Tone.js and @tonejs/midi
- **Video Preview**: React Player for video playback and composition preview
- **Entry Point**: [`index.html`](index.html) → React app bootstrapped in [`src/`](src/)

### Backend (Node.js + Python Hybrid)

- **Node.js Server** ([`backend/server.js`](backend/server.js)): Express.js API handling HTTP requests, auth, CORS, and routing
- **Job Queue** ([`backend/services/queueService.js`](backend/services/queueService.js)): Bull (Redis-backed) queue that runs composition jobs asynchronously and reports progress
- **Python Video Processing** ([`backend/utils/video_processor.py`](backend/utils/video_processor.py)): GPU-accelerated video composition, invoked by the queue worker as a subprocess with `--midi-json`, `--video-files-json`, and `--output-path` arguments. This wraps [`backend/python/video_composer.py`](backend/python/video_composer.py), the core chunk-based compositor.
- **API Routes** (mounted in [`backend/server.js`](backend/server.js)):
  - `/api/auth` — registration, login, session (JWT)
  - `/api/billing` — Stripe subscription status/checkout/portal
  - `/api/projects` — saved projects, clips, collaborator invites, render status/export
  - `/api/midi` — MIDI file parsing and note extraction
  - `/api/video` — video upload and metadata extraction
  - `/api/process-videos` — **the actual composition pipeline entry point** (submit job, poll `/progress/:jobId`, `/status/:jobId`, `/result/:jobId`)
  - `/api/autotune` (+ `/api/autotune/precache`) — pitch-correction processing and precaching
  - `/api/upload` — file upload handling with large payload support
  - `/api/compose` — **deprecated**; every endpoint in this route now returns a deprecation response. Kept only for backward compatibility with any old clients. Use `/api/process-videos` instead.

> `backend/routes/shareRoutes.js` and `backend/routes/socialRoutes.js` still exist in the codebase from an earlier social-feed feature but are **not mounted** in `server.js` and are no longer part of the product. That feature was deliberately dropped: a public feed where users could publish compositions meaningfully increases exposure to third-party copyright claims (MIDI transcriptions and video clips can both encode copyrighted material), which isn't a risk worth carrying for a free/low-revenue product. See [`docs/CHANGELOG.md`](docs/CHANGELOG.md) for the full note, and [Legal](#legal) for how the private-dashboard-only model is reflected in the ToS/DMCA policy.
>
> **Recommended cleanup** (dead code implementing a dropped feature is a maintenance and confusion risk — e.g., an "AI Tool Integration Guide" reader could pattern-match against it and resurrect the feature by accident):
> - Delete `backend/routes/shareRoutes.js`, `backend/routes/socialRoutes.js`, `backend/controllers/shareController.js`, `backend/controllers/socialController.js`
> - Delete `src/components/Social/SocialFeed.jsx`, `CompositionDetail.jsx`, `CompositionCard.jsx`, `ShareCompositionModal.jsx`, `UserProfile.jsx`, `Notifications.jsx`
> - **Careful:** `src/components/Social/Settings.jsx` and `src/components/Social/Social.css` are *not* part of the dropped feature — both are actively imported in `App.jsx`. Move them out of the `Social/` folder (e.g. into `src/components/` directly) before deleting the rest of the directory, and update the two import paths in `App.jsx` accordingly.
> - Remove the `S3_BUCKET`/`SHARE_URL_EXPIRY` env vars from `.env`/`.env.example` if nothing else picks up `@aws-sdk/client-s3` usage (see the note in Technologies Stack above)

### Technologies Stack

**Frontend:**

- React 18.3, Vite 5.4
- Tone.js (MIDI playback), @tonejs/midi (MIDI parsing)
- Axios (HTTP client), React Player (video playback)
- @dnd-kit (drag-and-drop), lucide-react (icons)

**Backend (Node.js):**

- Express.js, CORS, midi-parser-js
- Bull + Redis (job queue)
- jsonwebtoken + bcryptjs (auth)
- Stripe (billing)
- @aws-sdk/client-s3 — currently only used by `shareController.js` (the removed share-link feature, see note below); if that file is deleted per the recommendation below, this dependency becomes unused unless/until it's picked back up for the separate R2 storage work described in [Deployment](#deployment)

**Backend (Python 3.10+):**

- PyTorch 2.7 (GPU tensors, CUDA 11.8 build), torchvision, torchaudio
- FFmpeg (via subprocess — NVENC when available, libx264 fallback)
- librosa, soundfile, scipy (audio analysis / pitch shifting)
- moviepy 2.1.x
- aubio (pitch/audio analysis), pyrubberband (optional time-stretching — requires the external `rubberband` binary on `PATH`)
- CREPE (optional, TensorFlow-based pitch detection) — **not installed by default**; code falls back to librosa-based pitch detection if it's absent. See [`backend/requirements.txt`](backend/requirements.txt) for why it's omitted by default.

> **Python version requirement:** `backend/python/ffmpeg_profiles.py` uses `bool | None` type-hint syntax (PEP 604), which requires **Python 3.10+**. The `>=3.8` requirement previously stated here (and in `setup.py`) was inaccurate — a 3.8/3.9 interpreter will fail on import.

## Project Structure

```
├── index.html                          # Entry point for React app
├── package.json                        # Frontend dependencies + npm scripts (dev/build/test)
├── vite.config.js                      # Vite build configuration
├── config.js                           # Application configuration (API URLs, paths)
│
├── src/                                # React frontend source
│   ├── components/                     # UI components (MIDI upload, video grid, etc.)
│   ├── hooks/                          # Custom React hooks
│   └── App.jsx                         # Main React application
│
├── backend/
│   ├── server.js                       # Express.js API server
│   ├── package.json                    # Backend Node.js dependencies
│   ├── requirements.txt                # Python dependencies (PyTorch, moviepy, etc.)
│   │
│   ├── routes/                         # Express route handlers (see Architecture above)
│   ├── controllers/                    # Request handlers (auth, billing, projects, sharing, ...)
│   ├── services/                       # Business logic — job queue, billing access, caching
│   ├── middleware/                     # Express middleware (auth, validation)
│   ├── db/                             # Database access
│   │
│   ├── python/
│   │   └── video_composer.py           # Core chunk-based GPU-accelerated video compositor
│   │
│   └── utils/                          # Shared Python + Node utilities (video_processor.py entrypoint,
│                                        # ffmpeg_gpu.py, ffmpeg_profiles.py, autotune.py, r2Client.js, ...)
│
├── public/                             # Static assets
├── docs/                               # Design notes, style guide, and archived fix/migration history
└── test_*.py, test_*.js, test-*.ps1    # Integration and unit tests
```

## Installation

### Prerequisites

**Required:**

- **Python 3.10+** (3.11 recommended — see version note above)
- **Node.js 18+** (project uses ESM `"type": "module"` throughout)
- **FFmpeg**, available on `PATH`
- **Redis** (for the Bull job queue — local install or a hosted instance)

**Optional (Highly Recommended):**

- **NVIDIA GPU** with a CUDA 11.8-compatible driver, for GPU-accelerated encoding (NVENC) and PyTorch tensor ops
- The external **`rubberband`** CLI binary on `PATH`, if you want `pyrubberband`'s time-stretching rather than the librosa fallback

### System Dependencies

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg redis-server python3-dev python3-pip nodejs npm
```

**Windows:**

- Install [Python](https://www.python.org/downloads/) 3.10+
- Install [Node.js](https://nodejs.org/) 18+
- Install [FFmpeg](https://ffmpeg.org/download.html) and add to PATH
- Install [Redis](https://github.com/microsoftarchive/redis/releases) or run it via WSL/Docker
- Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- (Optional) Install a CUDA 11.8-compatible [NVIDIA driver](https://www.nvidia.com/drivers/)

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

4. **Set up a Python virtual environment and install dependencies:**

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cd ..
```

`requirements.txt` includes a PyTorch CUDA 11.8 wheel index. If your GPU/driver needs a different CUDA build, get the matching install command from [pytorch.org](https://pytorch.org/get-started/locally/) before running the install above.

5. **Verify GPU setup (optional but recommended):**

```bash
cd backend
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
cd ..
```

### Environment Configuration

Copy [`.env.example`](.env.example) to `.env` in the project root and fill in the values for your environment. See that file for the full list of required and optional variables (Redis connection, JWT secret, Stripe keys, S3/R2 storage, etc.).

## Running the Application

### Development

You'll need Redis running locally (or reachable via your `.env` config) before starting the backend, since the job queue depends on it.

**Terminal 1 — Backend:**

```bash
cd backend
npm start
# Express server starts; port is configured via .env (see .env.example)
```

**Terminal 2 — Frontend:**

```bash
npm run dev
# Vite dev server on http://localhost:5173
```

### Production Build

```bash
npm run build     # generates dist/
npm run preview   # serve the production build locally
```

For actual production deployment (not just a local production build), see [Deployment](#deployment) below.

## Composition Pipeline — How It Actually Works

1. Frontend submits a job to `POST /api/process-videos` with MIDI note data and video clip references.
2. The request is enqueued via Bull (`backend/services/queueService.js`), backed by Redis.
3. The queue worker spawns `backend/utils/video_processor.py` as a subprocess, passing MIDI/video data via temp JSON files (`--midi-json`, `--video-files-json`) and an output path.
4. `video_processor.py` drives `backend/python/video_composer.py`, which composites video chunks per note event — using GPU tensor ops and NVENC where available, falling back to CPU/libx264 otherwise.
5. The client polls `GET /api/process-videos/progress/:jobId` (or `/status/:jobId`) for progress, then fetches the result via `/result/:jobId`.

The legacy `/api/compose/*` endpoints are deprecated stubs kept only for backward compatibility — don't build new features against them.

## Testing

```bash
npm run test:regressions   # python -m unittest -v test_video_composer_regressions
npm run test:components    # python test_components.py — validates GPU setup, processing utils, autotune, composer
npm run test:quick         # test-quick.ps1 — basic API connectivity (Windows/PowerShell)
npm run test:dev           # test-dev.ps1
npm run test:e2e           # test-e2e.ps1 — full pipeline validation
npm run test:production    # test-production.ps1
```

The PowerShell-based test scripts (`test:quick`, `test:dev`, `test:e2e`, `test:production`) require `pwsh` and are Windows-oriented; the Python-based ones (`test:regressions`, `test:components`) run anywhere.

## Deployment

This project currently runs as a monolith: the Node API and the Python worker share one host, since `queueService.js` spawns the Python process locally rather than dispatching it to a separate worker. For a low-traffic/free-tier deployment, the recommended path is a single scale-to-zero container (e.g. Modal or RunPod Serverless) running both Node and Python together, with:

- **Frontend** on Vercel
- **Redis** via a serverless provider (e.g. Upstash) rather than an always-on instance
- **File storage** on Cloudflare R2 (S3-compatible — see `backend/utils/r2Client.js`)

A fully decoupled architecture (separate always-on API + scale-to-zero GPU worker function) is a follow-up step once there's real usage data — see `modal_app.py` at the repo root for a starting point on the GPU-worker side.

## Legal

- [Terms of Service](docs/legal/terms-of-service.md)
- [DMCA / Copyright Policy](docs/legal/dmca-policy.md)

User-uploaded content (MIDI and video) is stored privately per-account — there is no social feed or public-sharing feature (see the note on removed features above and [`docs/CHANGELOG.md`](docs/CHANGELOG.md)). See the linked policies for what this does and doesn't mean for copyright responsibility — this project does not currently have a `LICENSE` file for its own source code, which should be added before any public release (see [Contributing](#contributing)).

## Troubleshooting

### GPU Not Detected

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False` on a machine with an NVIDIA GPU, check that the installed torch build's CUDA version is compatible with your driver (see the CUDA 11.8 note in Installation above), and that the GPU driver itself is installed.

### FFmpeg Encoding Errors

```bash
ffmpeg -version
ffmpeg -i input.mp4 -c:v libx264 -preset fast output.mp4
```

Windows: ensure FFmpeg's `bin` folder is on your System PATH.

### Memory Errors (OOM)

**GPU memory:** lower the memory-limit setting passed into `EnhancedVideoProcessor` (see `backend/utils/video_processor.py` for where this is configured), and clear the CUDA cache between runs:

```bash
python -c "import torch; torch.cuda.empty_cache()"
```

**Node.js heap:** increase the memory limit before starting the server:

```bash
node --max-old-space-size=8192 backend/server.js
```

### Redis Connection Errors

**Symptoms:** job queue never processes, or errors referencing `ioredis`/Bull on startup.

**Solutions:** confirm Redis is running and reachable at the host/port/password/db configured in `.env` (see `.env.example`); `backend/services/cacheService.js` and `backend/services/queueService.js` both read `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `REDIS_DB`.

### CORS Errors in Browser Console

Check the CORS/allowed-origins configuration in `backend/server.js` and confirm `ALLOWED_ORIGINS`/`FRONTEND_URL` in `.env` matches the URL you're actually loading the frontend from.

### MIDI File Not Parsing

1. Confirm the file is a Standard MIDI File, not a proprietary/DAW-specific format.
2. Test with a simple, known-good MIDI file first.
3. Check logs from `backend/routes/midiRoutes.js` for the specific parse error.

## Project History

This project has gone through several rounds of architectural refactoring (chunk-based video processing migration, GPU acceleration fixes, note-triggered composition rework, and the removal of the social/sharing feature). See [`docs/CHANGELOG.md`](docs/CHANGELOG.md) for a consolidated, deduplicated history — it replaces what used to be nine overlapping status-report files under `docs/archive/`. If you're looking for *why* something is built the way it is, check there; if you're looking for *how it currently works*, this file and the code are the source of truth.

## Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

**Key areas for contribution:**

- Additional video effects and filters
- Alternative autotune algorithms
- UI/UX improvements
- Performance optimizations
- Cross-platform compatibility

**Note:** this repository does not currently include a `LICENSE` file. Until one is added, the default legal position is that all rights are reserved — worth resolving before accepting external contributions or a public launch.

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
- **Async job queue**: Composition jobs run through Bull/Redis, not a synchronous request — always check `queueService.js` before assuming a request/response is instantaneous
- **GPU Acceleration**: PyTorch CUDA tensors + NVENC where available; both have CPU/libx264 fallbacks — don't assume GPU is present
- **Hybrid Architecture**: Node.js handles HTTP/routing/auth/billing/queueing; Python (invoked as a subprocess) handles GPU/video computation
- **Large File Handling**: Videos can be 100MB+, requiring streaming/large-payload handling in upload routes

### Common Modification Patterns

**When asked to modify video processing:**

- **Files**: [`backend/python/video_composer.py`](backend/python/video_composer.py) (core compositor), [`backend/utils/video_processor.py`](backend/utils/video_processor.py) (subprocess entrypoint/CLI args)
- **Pattern**: GPU tensor ops use `torch.cuda`; encoding goes through `backend/python/ffmpeg_profiles.py` for consistent quality settings across call sites
- **Testing**: `npm run test:regressions` and `npm run test:components`

**When asked to modify MIDI handling:**

- **File**: [`backend/routes/midiRoutes.js`](backend/routes/midiRoutes.js)
- **Pattern**: Uses `midi-parser-js` for parsing

**When asked to modify the composition API:**

- **File**: [`backend/routes/processVideos.js`](backend/routes/processVideos.js) — this is the live endpoint, **not** `backend/routes/composition.js` (deprecated stub)
- **Pattern**: Job submission + polling (`/progress/:jobId`, `/status/:jobId`, `/result/:jobId`), not a single synchronous call

**When asked to modify auth/billing/projects:**

- **Files**: `backend/routes/authRoutes.js`, `backend/routes/billingRoutes.js`, `backend/routes/projectRoutes.js` + matching `controllers/`
- **Pattern**: JWT auth via `authenticateToken` middleware; Stripe client is created lazily and returns `null` if `STRIPE_SECRET_KEY` isn't set (billing routes no-op in that case)

**When asked to modify UI:**

- **Files**: [`src/`](src/) React components
- **Pattern**: Axios for API calls, React hooks for state
- **Testing**: Manual testing via `npm run dev`
