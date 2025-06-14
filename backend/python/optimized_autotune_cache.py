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
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class OptimizedAutotuneCache:
    """
    High-performance autotune cache system for instrument videos.
    
    This system pre-processes instrument videos to all required MIDI notes,
    creating a cached library of tuned videos that can be instantly retrieved
    during composition.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_workers: int = 4):
        """
        Initialize the optimized autotune cache.
        
        Args:
            cache_dir: Directory to store cached tuned videos
            max_workers: Number of parallel workers for autotune processing
        """
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), 'autotunesyncer_cache')
        self.max_workers = max_workers
        self.cache_index = {}
        self.processing_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_time': 0.0,
            'videos_processed': 0
        }
        self.lock = threading.RLock()
        
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
        """Save cache index to disk"""
        index_file = os.path.join(self.cache_dir, 'cache_index.json')
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save cache index: {e}")

    def _generate_video_hash(self, video_path: str) -> str:
        """Generate hash for video file to use as cache key"""
        try:
            with open(video_path, 'rb') as f:
                # Read first and last 1KB to create fingerprint
                start_data = f.read(1024)
                f.seek(-1024, 2)
                end_data = f.read(1024)
                
            file_stats = os.stat(video_path)
            hash_input = f"{start_data}{end_data}{file_stats.st_size}{file_stats.st_mtime}"
            return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        except Exception as e:
            logging.warning(f"Failed to hash video {video_path}: {e}")
            return hashlib.md5(video_path.encode()).hexdigest()[:16]

    def _get_cache_key(self, video_path: str, midi_note: int) -> str:
        """Generate cache key for video/note combination"""
        video_hash = self._generate_video_hash(video_path)
        return f"{video_hash}_{midi_note}"

    def _get_cached_video_path(self, cache_key: str) -> str:
        """Get path where cached video should be stored"""
        return os.path.join(self.cache_dir, f"{cache_key}.mp4")

    def get_tuned_video(self, video_path: str, midi_note: int) -> Optional[str]:
        """
        Get cached tuned video or create it if not exists.
        
        Args:
            video_path: Path to original instrument video
            midi_note: Target MIDI note for tuning
            
        Returns:
            Path to tuned video or None if failed
        """
        with self.lock:
            cache_key = self._get_cache_key(video_path, midi_note)
            cached_path = self._get_cached_video_path(cache_key)
            
            # Check if cached version exists and is valid
            if cache_key in self.cache_index and os.path.exists(cached_path):
                # Verify cached file is not corrupted
                if os.path.getsize(cached_path) > 1000:  # Basic size check
                    self.processing_stats['cache_hits'] += 1
                    logging.info(f"✅ Cache HIT: {os.path.basename(video_path)} → MIDI {midi_note}")
                    return cached_path
                else:
                    # Remove corrupted cache entry
                    logging.warning(f"Removing corrupted cache entry: {cache_key}")
                    self.cache_index.pop(cache_key, None)
                    if os.path.exists(cached_path):
                        os.remove(cached_path)
            
            # Cache miss - need to create tuned video
            self.processing_stats['cache_misses'] += 1
            logging.info(f"❌ Cache MISS: {os.path.basename(video_path)} → MIDI {midi_note} (processing...)")
            
            tuned_path = self._create_tuned_video(video_path, midi_note, cached_path)
            if tuned_path:
                # Update cache index
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
            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp(prefix='autotune_processing_')
              # Extract audio from video
            audio_path = os.path.join(temp_dir, 'audio.wav')
            cmd = [
                'ffmpeg', '-y', '-i', video_path, 
                '-vn', '-acodec', 'pcm_s16le', 
                '-ac', '1', '-ar', '44100', audio_path
            ]
            subprocess.run(cmd, check=True, capture_output=True, 
                         text=True, encoding='utf-8', errors='replace')
              # Autotune audio
            autotuned_audio_path = os.path.join(temp_dir, 'autotuned.wav')
            autotune_script = os.path.join(os.path.dirname(__file__), 'autotune.py')
            cmd = ['python', autotune_script, audio_path, autotuned_audio_path, str(midi_note)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, 
                                  encoding='utf-8', errors='replace')
              # Combine original video with autotuned audio
            cmd = [
                'ffmpeg', '-y', 
                '-i', video_path,           # Original video
                '-i', autotuned_audio_path, # Autotuned audio
                '-c:v', 'copy',             # Copy video stream
                '-c:a', 'aac',              # Encode audio to AAC
                '-map', '0:v:0',            # Video from first input
                '-map', '1:a:0',            # Audio from second input
                '-shortest',                # Match shortest stream
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True, 
                         text=True, encoding='utf-8', errors='replace')
            
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

    def preprocess_composition(self, midi_data: Dict, video_paths: Dict[str, str]) -> Dict[str, Dict[int, str]]:
        """
        Pre-process all required note combinations for a composition.
        
        This is the key optimization - we analyze the MIDI data to find all
        unique note/instrument combinations, then batch process them in parallel.
        
        Args:
            midi_data: MIDI composition data
            video_paths: Mapping of instrument names to video file paths
            
        Returns:
            Mapping of instrument -> {midi_note -> tuned_video_path}
        """
        logging.info("🚀 Starting optimized composition preprocessing...")
        start_time = time.time()
        
        # Analyze MIDI data to find all required note/instrument combinations
        required_combinations = self._analyze_midi_requirements(midi_data, video_paths)
        
        total_combinations = sum(len(notes) for notes in required_combinations.values())
        logging.info(f"📊 Found {total_combinations} unique note/instrument combinations to process")
        
        if total_combinations == 0:
            logging.warning("No note/instrument combinations found!")
            return {}
        
        # Process all combinations in parallel
        tuned_videos = {}
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all processing tasks
            future_to_info = {}
            for instrument, midi_notes in required_combinations.items():
                if instrument not in video_paths:
                    logging.warning(f"No video found for instrument: {instrument}")
                    continue
                    
                video_path = video_paths[instrument]
                tuned_videos[instrument] = {}
                
                for midi_note in midi_notes:
                    future = executor.submit(self.get_tuned_video, video_path, midi_note)
                    future_to_info[future] = (instrument, midi_note)
            
            # Collect results as they complete
            for future in as_completed(future_to_info):
                instrument, midi_note = future_to_info[future]
                completed += 1
                
                try:
                    tuned_path = future.result()
                    if tuned_path:
                        tuned_videos[instrument][midi_note] = tuned_path
                        logging.info(f"✅ [{completed}/{total_combinations}] {instrument} → MIDI {midi_note}")
                    else:
                        logging.error(f"❌ [{completed}/{total_combinations}] Failed: {instrument} → MIDI {midi_note}")
                except Exception as e:
                    logging.error(f"❌ [{completed}/{total_combinations}] Error processing {instrument} → MIDI {midi_note}: {e}")
        
        processing_time = time.time() - start_time
        successful = sum(len(notes) for notes in tuned_videos.values())
        
        logging.info(f"🎉 Preprocessing complete!")
        logging.info(f"   ✅ Successfully processed: {successful}/{total_combinations}")
        logging.info(f"   ⏱️  Total time: {processing_time:.2f}s")
        logging.info(f"   📊 Average per combination: {processing_time/max(1, successful):.2f}s")
        
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
