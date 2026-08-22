# Symphovie — Development History

This consolidates what used to be nine separate, overlapping status reports in `docs/archive/` into one chronological record. Several of those files documented the exact same event from different sessions (four separate "migration complete" reports on the same May 28, 2025 change, for example) — this file keeps one entry per real change instead.

If you're looking for *how the system currently works*, see the [README](../README.md) — this file is historical context for *why* it works that way, not current documentation.

---

## Note-triggered video sequence integration

Integrated `_create_note_triggered_video_sequence` into the main composition pipeline so videos synchronize with individual MIDI notes instead of playing continuously. Added FFmpeg filter-complex construction for note-by-note playback, pitch adjustment via the `asetrate` filter, note-timing sync via `enable='between(t,start,end)'`, and `amix`-based audio mixing for simultaneous notes.

## Chunk-based video processing migration — May 28, 2025

**Problem:** Note-by-note processing created one FFmpeg process per MIDI note — a 136-note file meant 136 individual processes, causing exponential resource usage, timeouts, and (critically) a silent-video bug where audio wasn't included in final output.

**Fix:** Migrated to chunk-based processing — grouping notes into batches (~34 processes for the same 136-note workload that previously spawned 136) and fixing the audio-stream handling that caused the silent-output bug.

**Result:** This is the architecture still in use today (`backend/python/video_composer.py`'s chunk-based compositor, invoked via `backend/utils/video_processor.py`).

## GPU preprocessing & MIDI sync fixes

- Fixed `gpu_subprocess_run` being called with incorrect parameters, which was causing crashes during GPU preprocessing.
- Added proper CPU fallback when GPU encoding fails (`ffmpeg_gpu_encode`).
- Added the `OptimizedAutotuneCache` class (threading + cache management) that `VideoComposer` depends on at init.
- Fixed a data-structure mismatch between the JavaScript preprocessing stage (which produced a `'video'` key) and the Python composition stage (which expected `'path'` or `'videoData'`), which was causing "No valid video data for track" errors.
- Fixed a video-output bug where composed output showed text instead of actual video content (FFmpeg command construction/stream-detection error).

## FFmpeg backend improvements

Added a codec-detection helper (`get_optimal_video_codec()` in `video_utils.py`) that checks for available hardware encoders in priority order (NVIDIA `h264_nvenc`, Intel `h264_qsv`, falling back to `libx264`). This is the basis for the GPU/CPU fallback behavior still in use.

## Social feed & sharing features — removed

The codebase at one point included `backend/routes/shareRoutes.js` and `backend/routes/socialRoutes.js`, intended to support a social feed where users could publish and share their compositions. These were never mounted in `server.js` and have since been removed as a deliberate decision: user-uploaded MIDI and video content can implicate third-party copyright (a MIDI transcription of a copyrighted song, a video containing a copyrighted recording), and a public social feed meaningfully increases exposure to copyright claims compared to the private-dashboard-only model the app uses today. See [Legal](../README.md#legal) in the README for the current data-handling model.

---

*Superseded files (kept in git history, removed from `docs/archive/`): `MIGRATION_COMPLETE_FINAL.md`, `MIGRATION_COMPLETION_REPORT.md`, `MIGRATION_FINAL_COMPLETE.md`, `MIGRATION_SUCCESS_REPORT.md`, `VIDEO_COMPOSITION_FIX_COMPLETE.md`, `VIDEO_COMPOSITION_FIX_SUMMARY.md`, `GPU_FIX_SUMMARY.md`, `FIXES_COMPLETE_SUMMARY.md`, `NOTE_TRIGGERED_INTEGRATION_COMPLETE.md`, `FFMPEG_IMPROVEMENTS.md`.*
