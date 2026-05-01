#!/usr/bin/env python3
"""
Optimized Autotune Cache System for AutoTune Syncer

This module implements an efficient note pre-caching system that dramatically improves
performance by pre-tuning instrument videos to all required MIDI notes once, then
reusing the cached tuned videos for composition.

Key Performance Improvements:
- Pre-tune each instrument video to all required notes once
- Cache tuned videos on disk for reuse across compositions  
- Eliminate redundant autotune processing during composition
- Reduce 5+ minute processing times to seconds
"""

import os
import sys
import json
import hashlib
import tempfile
import logging
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Optional in-process audio dependencies (enable batch path; fall back to subprocess if missing)
try:
    import numpy as np
    import soundfile as sf
    _AUDIO_LIBS_AVAILABLE = True
except ImportError:
    _AUDIO_LIBS_AVAILABLE = False

# Lazily import autotune helpers only when the batch path is active
_autotune_detect_pitch = None
_autotune_process_predetected = None

def _ensure_autotune_imports():
    """Import pitch-detection and shift helpers from autotune.py once."""
    global _autotune_detect_pitch, _autotune_process_predetected
    if _autotune_detect_pitch is not None:
        return True
    try:
        _backend_python = os.path.dirname(os.path.abspath(__file__))
        if _backend_python not in sys.path:
            sys.path.insert(0, _backend_python)
        from autotune import _detect_fundamental_pitch, process_audio_predetected
        _autotune_detect_pitch = _detect_fundamental_pitch
        _autotune_process_predetected = process_audio_predetected
        return True
    except Exception as e:
        logging.warning(f"[precache] Cannot import autotune helpers for batch path: {e}")
        return False

class OptimizedAutotuneCache:
    """
    High-performance autotune cache system for instrument videos.
    
    This system pre-processes instrument videos to all required MIDI notes,
    creating a cached library of tuned videos that can be instantly retrieved
    during composition.
    """

    # Bump when autotune algorithm or encoding settings change to force cache regeneration.
    CACHE_VERSION = "v2"

    def __init__(self, cache_dir: Optional[str] = None, max_workers: int = 4):
        """
        Initialize the optimized autotune cache.
        
        Args:
            cache_dir: Directory to store cached tuned videos
            max_workers: Number of parallel workers for autotune processing
        """
        default_cache = str(Path.home() / '.autotunesyncer' / 'cache_v2')
        self.cache_dir = cache_dir or default_cache
        self.max_workers = max_workers
        self.cache_index = {}
        self.processing_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_time': 0.0,
            'videos_processed': 0
        }
        self.lock = threading.RLock()          # guards cache_index and stats only
        self._key_locks: Dict[str, threading.Lock] = {}  # per-key lock for parallel processing
        self._key_locks_lock = threading.Lock()  # guards _key_locks dict
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing cache index
        self._load_cache_index()
        
        logging.info(f"Optimized autotune cache initialized at: {self.cache_dir}")
        logging.info(f"Cache contains {len(self.cache_index)} pre-tuned videos")

    def _load_cache_index(self):
        """Load cache index from disk"""
        index_file = os.path.join(self.cache_dir, 'cache_index.json')
        if os.path.exists(index_file):
            try:
                with open(index_file, 'r') as f:
                    self.cache_index = json.load(f)
                logging.info(f"Loaded cache index with {len(self.cache_index)} entries")
            except Exception as e:
                logging.warning(f"Failed to load cache index: {e}")
                self.cache_index = {}
        else:
            self.cache_index = {}

    def _save_cache_index(self):
        """Save cache index to disk using an atomic write to prevent corruption
        if multiple processes write simultaneously."""
        index_file = os.path.join(self.cache_dir, 'cache_index.json')
        tmp_file = index_file + '.tmp'
        try:
            with open(tmp_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
            os.replace(tmp_file, index_file)  # atomic on POSIX and Windows
        except Exception as e:
            logging.error(f"Failed to save cache index: {e}")

    def _generate_video_hash(self, video_path: str) -> str:
        """Generate a content-based hash for a video file.

        Intentionally excludes mtime so that the same video content uploaded
        to different paths (or at different times) produces the same key,
        enabling pre-cache warm-ups to benefit subsequent composition uploads.
        """
        try:
            with open(video_path, 'rb') as f:
                start_data = f.read(1024)
                f.seek(-min(1024, os.path.getsize(video_path)), 2)
                end_data = f.read(1024)
            file_size = os.path.getsize(video_path)
            # Content fingerprint only — no mtime
            hash_input = start_data + end_data + str(file_size).encode()
            return hashlib.sha256(hash_input).hexdigest()[:16]
        except Exception as e:
            logging.warning(f"Failed to hash video {video_path}: {e}")
            return hashlib.md5(video_path.encode()).hexdigest()[:16]

    def _get_cache_key(self, video_path: str, midi_note: int) -> str:
        """Generate cache key for video/note combination"""
        video_hash = self._generate_video_hash(video_path)
        return f"{self.CACHE_VERSION}_{video_hash}_{midi_note}"

    def _get_cached_video_path(self, cache_key: str) -> str:
        """Get path where cached video should be stored"""
        return os.path.join(self.cache_dir, f"{cache_key}.mp4")

    def _get_key_lock(self, cache_key: str) -> threading.Lock:
        """Return (creating if needed) the per-key lock for a cache entry."""
        with self._key_locks_lock:
            if cache_key not in self._key_locks:
                self._key_locks[cache_key] = threading.Lock()
            return self._key_locks[cache_key]

    def get_tuned_video(self, video_path: str, midi_note: int) -> Optional[str]:
        """
        Get cached tuned video or create it if not exists.

        Uses per-key locking so that parallel preprocessing for different
        note/instrument combinations runs concurrently.
        """
        cache_key = self._get_cache_key(video_path, midi_note)
        cached_path = self._get_cached_video_path(cache_key)

        # Fast path: already cached
        with self.lock:
            if cache_key in self.cache_index and os.path.exists(cached_path):
                if os.path.getsize(cached_path) > 1000:
                    self.processing_stats['cache_hits'] += 1
                    logging.info(f"✅ Cache HIT: {os.path.basename(video_path)} → MIDI {midi_note}")
                    return cached_path
                # Corrupted entry — evict under the global lock
                logging.warning(f"Removing corrupted cache entry: {cache_key}")
                self.cache_index.pop(cache_key, None)
                if os.path.exists(cached_path):
                    os.remove(cached_path)

        # Slow path: acquire per-key lock to avoid duplicate work
        key_lock = self._get_key_lock(cache_key)
        with key_lock:
            # Double-check after acquiring per-key lock
            with self.lock:
                if cache_key in self.cache_index and os.path.exists(cached_path):
                    if os.path.getsize(cached_path) > 1000:
                        self.processing_stats['cache_hits'] += 1
                        return cached_path

            with self.lock:
                self.processing_stats['cache_misses'] += 1
            logging.info(f"❌ Cache MISS: {os.path.basename(video_path)} → MIDI {midi_note} (processing...)")

            tuned_path = self._create_tuned_video(video_path, midi_note, cached_path)
            if tuned_path:
                with self.lock:
                    self.cache_index[cache_key] = {
                        'original_video': video_path,
                        'midi_note': midi_note,
                        'cached_path': cached_path,
                        'created_at': time.time(),
                        'file_size': os.path.getsize(cached_path)
                    }
                    self._save_cache_index()
                return tuned_path

        return None

    def _create_tuned_video(self, video_path: str, midi_note: int, output_path: str) -> Optional[str]:
        """
        Create tuned video using autotune processing.
        
        Args:
            video_path: Original video path
            midi_note: Target MIDI note
            output_path: Where to save tuned video
            
        Returns:
            Path to created video or None if failed
        """
        start_time = time.time()
        temp_dir = None
        
        try:
            # Ensure output directory exists (may have been deleted externally)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp(prefix='autotune_processing_')

            # ── Step 1: extract audio ─────────────────────────────────────────
            audio_path = os.path.join(temp_dir, 'audio.wav')
            cmd = [
                'ffmpeg', '-y', '-i', video_path, 
                '-vn', '-acodec', 'pcm_s16le', 
                '-ac', '1', '-ar', '44100', audio_path
            ]
            r1 = subprocess.run(cmd, capture_output=True, text=True,
                                encoding='utf-8', errors='replace')
            if r1.returncode != 0:
                logging.error(f"[precache] Audio extraction failed (rc={r1.returncode}) "
                              f"for MIDI {midi_note}:\n{r1.stderr[-400:]}")
                return None

            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
                logging.error(f"[precache] Extracted audio is empty for MIDI {midi_note} "
                              f"— source video may have no audio stream")
                return None

            # ── Step 2: autotune audio ────────────────────────────────────────
            autotuned_audio_path = os.path.join(temp_dir, 'autotuned.wav')
            autotune_script = os.path.join(os.path.dirname(__file__), 'autotune.py')
            cmd = ['python', autotune_script, audio_path, autotuned_audio_path, str(midi_note)]
            r2 = subprocess.run(cmd, capture_output=True, text=True,
                                encoding='utf-8', errors='replace')
            if r2.returncode != 0:
                logging.error(f"[precache] Autotune script failed (rc={r2.returncode}) "
                              f"for MIDI {midi_note}:\n{r2.stderr[-400:]}")
                return None

            if not os.path.exists(autotuned_audio_path) or os.path.getsize(autotuned_audio_path) < 100:
                logging.error(f"[precache] Autotuned audio is empty for MIDI {midi_note}")
                return None

            # ── Step 3: mux video + autotuned audio ───────────────────────────
            # Always re-encode video: browser MediaRecorder produces WebM/VP8 which
            # cannot be stream-copied into an MP4 container.
            from ffmpeg_profiles import get_video_encode_args
            cmd = [
                'ffmpeg', '-y', 
                '-i', video_path,           # Original video (may be WebM renamed .mp4)
                '-i', autotuned_audio_path, # Autotuned audio
                *get_video_encode_args(mode='master', use_gpu=False),  # CPU master quality
                '-c:a', 'aac',              # Encode audio to AAC
                '-map', '0:v:0',            # Video from first input
                '-map', '1:a:0',            # Audio from second input
                '-shortest',                # Match shortest stream
                output_path
            ]
            r3 = subprocess.run(cmd, capture_output=True, text=True,
                                encoding='utf-8', errors='replace')
            if r3.returncode != 0:
                logging.error(f"[precache] Mux failed (rc={r3.returncode}) "
                              f"for MIDI {midi_note}:\n{r3.stderr[-400:]}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                return None
            
            processing_time = time.time() - start_time
            self.processing_stats['processing_time'] += processing_time
            self.processing_stats['videos_processed'] += 1
            
            logging.info(f"✅ Created tuned video: MIDI {midi_note} in {processing_time:.2f}s")
            return output_path
            
        except Exception as e:
            logging.error(f"Failed to create tuned video for MIDI {midi_note}: {e}")
            # Clean up failed output
            if os.path.exists(output_path):
                os.remove(output_path)
            return None
        finally:
            # Clean up temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logging.warning(f"Failed to clean up temp dir: {e}")

    # ── Batch in-process path ──────────────────────────────────────────────────

    def _extract_source_audio(self, video_path: str, temp_dir: str) -> Optional[str]:
        """Extract audio from video to WAV once for batch reuse."""
        wav_path = os.path.join(temp_dir, 'source_audio.wav')
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '44100', wav_path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode != 0 or not os.path.exists(wav_path) or os.path.getsize(wav_path) < 100:
            logging.error(f"[precache] Source audio extraction failed: {r.stderr[-300:]}")
            return None
        return wav_path

    def _create_tuned_video_inproc(self, video_path: str, midi_note: int, output_path: str,
                                    mono_audio, sr: int, detected_pitch: float,
                                    original_gated_rms: float) -> Optional[str]:
        """
        Create one tuned video using pre-loaded audio and pre-detected pitch.
        Avoids repeated FFmpeg audio extraction and librosa.pyin calls.
        """
        temp_dir = None
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            temp_dir = tempfile.mkdtemp(prefix='autotune_inproc_')
            shifted_wav = os.path.join(temp_dir, f'shifted_{midi_note}.wav')

            # Reconstruct 2D audio array for process_audio_predetected
            audio_2d = np.expand_dims(mono_audio, axis=1).astype(np.float32)

            processed, _, _ = _autotune_process_predetected(
                audio_2d, sr, detected_pitch, midi_note,
                _mono_audio=mono_audio.copy(),
                _original_gated_rms=original_gated_rms,
            )
            sf.write(shifted_wav, processed, sr, subtype='PCM_16', format='WAV')

            # Mux shifted audio back with video — use master quality profile
            from ffmpeg_profiles import get_video_encode_args
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', shifted_wav,
                *get_video_encode_args(mode='master', use_gpu=False),  # CPU master (intermediate file)
                '-c:a', 'aac',
                '-map', '0:v:0', '-map', '1:a:0',
                '-shortest', output_path
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if r.returncode != 0:
                logging.error(f"[precache] Mux failed for MIDI {midi_note}: {r.stderr[-300:]}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                return None
            return output_path
        except Exception as e:
            logging.error(f"[precache] In-process tuning failed for MIDI {midi_note}: {e}")
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
            return None
        finally:
            if temp_dir and os.path.exists(temp_dir):
                try:
                    import shutil; shutil.rmtree(temp_dir)
                except Exception:
                    pass

    def _get_tuned_videos_batch(self, video_path: str, midi_notes: List[int]) -> Dict[int, str]:
        """
        Warm the cache for *midi_notes* from a single *video_path*.

        Extracts audio once and detects pitch once, then fans out pitch-shift
        work across the thread pool.  Falls back to the subprocess path per-note
        when the in-process audio libraries are unavailable or extraction fails.

        Returns a dict of {midi_note: cached_video_path} for all successful notes.
        """
        results: Dict[int, str] = {}

        # Separate into cache-hit and cache-miss notes
        miss_notes = []
        for midi_note in midi_notes:
            cache_key = self._get_cache_key(video_path, midi_note)
            cached_path = self._get_cached_video_path(cache_key)
            with self.lock:
                if cache_key in self.cache_index and os.path.exists(cached_path) and os.path.getsize(cached_path) > 1000:
                    self.processing_stats['cache_hits'] += 1
                    results[midi_note] = cached_path
                    continue
            miss_notes.append(midi_note)

        if not miss_notes:
            return results

        batch_ok = _AUDIO_LIBS_AVAILABLE and _ensure_autotune_imports()

        if batch_ok:
            # ── Batch path: extract audio + detect pitch once ──────────────
            temp_dir = tempfile.mkdtemp(prefix='autotune_batch_')
            try:
                wav_path = self._extract_source_audio(video_path, temp_dir)
                if wav_path:
                    audio, sr = sf.read(wav_path)
                    if audio.ndim == 1:
                        mono = audio.astype(np.float32)
                    else:
                        mono = np.mean(audio, axis=1).astype(np.float32)
                    detected_pitch = _autotune_detect_pitch(mono, sr)
                    import math
                    from autotune import _compute_gated_rms
                    original_gated_rms = _compute_gated_rms(mono, sr)
                    logging.info(f"[precache] Batch: detected pitch {detected_pitch:.1f} Hz "
                                 f"for {os.path.basename(video_path)}, "
                                 f"shifting to {len(miss_notes)} notes")

                    def _process_one_inproc(midi_note):
                        cache_key = self._get_cache_key(video_path, midi_note)
                        cached_path = self._get_cached_video_path(cache_key)
                        key_lock = self._get_key_lock(cache_key)
                        with key_lock:
                            with self.lock:
                                if cache_key in self.cache_index and os.path.exists(cached_path) and os.path.getsize(cached_path) > 1000:
                                    self.processing_stats['cache_hits'] += 1
                                    return midi_note, cached_path
                            with self.lock:
                                self.processing_stats['cache_misses'] += 1
                            tuned = self._create_tuned_video_inproc(
                                video_path, midi_note, cached_path,
                                mono, sr, detected_pitch, original_gated_rms)
                            if tuned:
                                with self.lock:
                                    self.cache_index[cache_key] = {
                                        'original_video': video_path,
                                        'midi_note': midi_note,
                                        'cached_path': cached_path,
                                        'created_at': time.time(),
                                        'file_size': os.path.getsize(cached_path),
                                    }
                                    self._save_cache_index()
                                return midi_note, tuned
                        return midi_note, None

                    with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                        for midi_note, path in pool.map(_process_one_inproc, miss_notes):
                            if path:
                                results[midi_note] = path
                            else:
                                # Note failed — log but don't abort others
                                logging.warning(f"[precache] Batch in-proc failed for MIDI {midi_note}")
                    return results
                # wav extraction failed — fall through to subprocess path
            except Exception as e:
                logging.warning(f"[precache] Batch path error: {e} — falling back to subprocess")
            finally:
                try:
                    import shutil; shutil.rmtree(temp_dir)
                except Exception:
                    pass

        # ── Subprocess fallback: one-by-one ───────────────────────────────
        logging.info(f"[precache] Using subprocess path for {len(miss_notes)} notes")
        for midi_note in miss_notes:
            path = self.get_tuned_video(video_path, midi_note)
            if path:
                results[midi_note] = path
        return results

    def preprocess_composition(self, midi_data: Dict, video_paths: Dict[str, str]) -> Dict[str, Dict[int, str]]:
        """
        Pre-process all required note combinations for a composition.

        Groups notes by source video and calls _get_tuned_videos_batch per
        instrument, so audio extraction and pitch detection happen once per
        source clip rather than once per note.
        """
        logging.info("🚀 Starting optimized composition preprocessing...")
        start_time = time.time()

        required_combinations = self._analyze_midi_requirements(midi_data, video_paths)
        total_combinations = sum(len(notes) for notes in required_combinations.values())
        logging.info(f"📊 Found {total_combinations} unique note/instrument combinations to process")

        if total_combinations == 0:
            logging.warning("No note/instrument combinations found!")
            return {}

        tuned_videos: Dict[str, Dict[int, str]] = {}

        def _process_instrument(instrument, notes_set):
            video_path = video_paths[instrument]
            return instrument, self._get_tuned_videos_batch(video_path, sorted(notes_set))

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(required_combinations))) as executor:
            future_to_instr = {
                executor.submit(_process_instrument, instr, notes): instr
                for instr, notes in required_combinations.items()
                if instr in video_paths
            }
            for future in as_completed(future_to_instr):
                try:
                    instrument, note_map = future.result()
                    tuned_videos[instrument] = note_map
                    logging.info(f"✅ {instrument}: {len(note_map)}/{len(required_combinations[instrument])} notes cached")
                except Exception as e:
                    instr = future_to_instr[future]
                    logging.error(f"❌ Error processing {instr}: {e}")

        processing_time = time.time() - start_time
        successful = sum(len(m) for m in tuned_videos.values())
        logging.info(f"🎉 Preprocessing complete! {successful}/{total_combinations} in {processing_time:.2f}s "
                     f"({processing_time/max(1, successful):.2f}s avg)")
        return tuned_videos


    def _analyze_midi_requirements(self, midi_data: Dict, video_paths: Dict[str, str]) -> Dict[str, Set[int]]:
        """
        Analyze MIDI data to find all unique instrument/note combinations.
        
        Args:
            midi_data: MIDI composition data
            video_paths: Available video paths
            
        Returns:
            Mapping of instrument -> set of required MIDI notes
        """
        requirements = {}
        
        for track in midi_data.get('tracks', []):
            instrument_name = track.get('instrument', {}).get('name', 'unknown')
            
            # Normalize instrument name to match video path keys
            normalized_name = self._normalize_instrument_name(instrument_name)
            
            # Only process if we have a video for this instrument
            if normalized_name not in video_paths:
                continue
                
            if normalized_name not in requirements:
                requirements[normalized_name] = set()
            
            # Collect all MIDI notes for this instrument
            for note in track.get('notes', []):
                midi_note = note.get('midi')
                if midi_note is not None:
                    requirements[normalized_name].add(midi_note)
        
        # Log analysis results
        for instrument, notes in requirements.items():
            logging.info(f"📝 {instrument}: {len(notes)} unique notes {sorted(list(notes))}")
        
        return requirements

    def _normalize_instrument_name(self, name: str) -> str:
        """Normalize instrument name to match video file naming convention"""
        return name.lower().replace(' ', '_').replace('-', '_')

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        with self.lock:
            total_requests = self.processing_stats['cache_hits'] + self.processing_stats['cache_misses']
            hit_rate = (self.processing_stats['cache_hits'] / max(1, total_requests)) * 100
            
            return {
                'cache_entries': len(self.cache_index),
                'cache_hits': self.processing_stats['cache_hits'],
                'cache_misses': self.processing_stats['cache_misses'], 
                'hit_rate_percent': round(hit_rate, 2),
                'total_processing_time': round(self.processing_stats['processing_time'], 2),
                'videos_processed': self.processing_stats['videos_processed'],
                'avg_processing_time': round(
                    self.processing_stats['processing_time'] / max(1, self.processing_stats['videos_processed']), 2
                ),
                'cache_directory': self.cache_dir
            }

    def cleanup_expired_cache(self, max_age_days: int = 30):
        """Clean up cache entries older than specified days"""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        expired_keys = []
        
        with self.lock:
            for cache_key, metadata in self.cache_index.items():
                if current_time - metadata.get('created_at', 0) > max_age_seconds:
                    expired_keys.append(cache_key)
            
            for cache_key in expired_keys:
                cached_path = self._get_cached_video_path(cache_key)
                if os.path.exists(cached_path):
                    os.remove(cached_path)
                self.cache_index.pop(cache_key, None)
            
            if expired_keys:
                self._save_cache_index()
                logging.info(f"Cleaned up {len(expired_keys)} expired cache entries")

# Utility function for easy integration
def create_optimized_cache(cache_dir: Optional[str] = None, max_workers: int = 4) -> OptimizedAutotuneCache:
    """Create and return an optimized autotune cache instance"""
    return OptimizedAutotuneCache(cache_dir=cache_dir, max_workers=max_workers)

# Test function for validation
def test_optimized_cache():
    """Test the optimized cache system"""
    print("🧪 Testing Optimized Autotune Cache System")
    
    cache = create_optimized_cache()
    
    # Test basic functionality
    test_video = "test.mp4"  # Would be real video in actual use
    test_notes = [60, 64, 67]  # C, E, G
    
    print(f"Cache directory: {cache.cache_dir}")
    print(f"Initial cache stats: {cache.get_cache_stats()}")
    
    # Simulate cache operations
    for note in test_notes:
        cache_key = cache._get_cache_key(test_video, note)
        print(f"Cache key for MIDI {note}: {cache_key}")
    
    print("✅ Optimized cache system initialized successfully!")

if __name__ == "__main__":
    test_optimized_cache()
