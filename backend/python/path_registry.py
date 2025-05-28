import os
import logging
import time
import threading
import weakref
from pathlib import Path
from typing import Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import psutil
from collections import OrderedDict

class CacheManager:
    """Enhanced cache manager with memory optimization"""
    
    def __init__(self, max_size: int = 1000, cleanup_interval: int = 300):
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self._cache = OrderedDict()
        self._access_times = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        
    def get(self, key: str) -> Optional[str]:
        """Get value from cache with LRU tracking"""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                self._access_times[key] = time.time()
                return value
        return None
    
    def set(self, key: str, value: str):
        """Set value in cache with automatic cleanup"""
        with self._lock:
            self._cache[key] = value
            self._access_times[key] = time.time()
            
            # Remove oldest items if cache is too large
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
            
            # Periodic cleanup
            if time.time() - self._last_cleanup > self.cleanup_interval:
                self._cleanup_expired()
    
    def _cleanup_expired(self, max_age: int = 3600):
        """Remove entries older than max_age seconds"""
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self._access_times.items()
            if current_time - access_time > max_age
        ]
        
        for key in expired_keys:
            if key in self._cache:
                del self._cache[key]
            del self._access_times[key]
        
        self._last_cleanup = current_time
        
        if expired_keys:
            logging.info(f"Cleaned up {len(expired_keys)} expired cache entries")

class PathRegistry:
    """Central registry for tracking file paths across the application with performance optimizations"""
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls):
        """Thread-safe singleton access method"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = PathRegistry()
        return cls._instance
    
    def __init__(self):
        self.drum_paths = {}          # Format: {"drum_name": "path/to/file.mp4"}
        self.instrument_paths = {}    # Format: {"instrument_name": {"note_X": "path.mp4"}}
        self.track_paths = {}         # Format: {"track_id": "path/to/directory"}
        self.debug_lookups = {}       # For tracking lookup attempts
        self.registry_file = None     # Path to save/load registry
        
        # Performance optimizations
        self._cache = CacheManager()
        self._path_validation_cache = {}
        self._stats = {
            'lookups': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'registrations': 0
        }
        self._lock = threading.RLock()
        
        # Weak references to avoid memory leaks
        self._weak_refs = weakref.WeakSet()
    
    def register_drum(self, drum_name: str, file_path: str, validate: bool = True):
        """Register a drum video path with optional validation"""
        with self._lock:
            norm_name = drum_name.lower().replace(' ', '_')
            
            if validate and not self._validate_path(file_path):
                logging.warning(f"Invalid drum path: {file_path}")
                return False
            
            self.drum_paths[norm_name] = file_path
            self._cache.set(f"drum:{norm_name}", file_path)
            self._stats['registrations'] += 1
            
            logging.info(f"Registered drum path: {norm_name} -> {file_path}")
            return True
        
    def register_instrument(self, instrument_name: str, note: str, file_path: str, validate: bool = True):
        """Register an instrument note video path with optional validation"""
        with self._lock:
            norm_name = instrument_name.lower().replace(' ', '_')
            
            if validate and not self._validate_path(file_path):
                logging.warning(f"Invalid instrument path: {file_path}")
                return False
            
            if norm_name not in self.instrument_paths:
                self.instrument_paths[norm_name] = {}
            
            self.instrument_paths[norm_name][str(note)] = file_path
            self._cache.set(f"instrument:{norm_name}:{note}", file_path)
            self._stats['registrations'] += 1
            
            logging.info(f"Registered instrument path: {norm_name}:{note} -> {file_path}")
            return True
    
    def register_track_directory(self, track_id: str, directory_path: str):
        """Register a track's base directory"""
        with self._lock:
            if not os.path.isdir(directory_path):
                logging.warning(f"Invalid track directory: {directory_path}")
                return False
            
            self.track_paths[track_id] = directory_path
            self._cache.set(f"track:{track_id}", directory_path)
            return True
        
    def get_drum_path(self, drum_key: str) -> Optional[str]:
        """Get path for drum video with caching"""
        with self._lock:
            self._stats['lookups'] += 1
            norm_name = drum_key.lower().replace(' ', '_')
            
            # Check cache first
            cache_key = f"drum:{norm_name}"
            cached_path = self._cache.get(cache_key)
            if cached_path:
                self._stats['cache_hits'] += 1
                return cached_path
            
            self._stats['cache_misses'] += 1
            
            # Direct lookup
            if norm_name in self.drum_paths:
                path = self.drum_paths[norm_name]
                self._cache.set(cache_key, path)
                return path
                
            # Check for partial matches
            for key in self.drum_paths:
                if norm_name in key or key in norm_name:
                    path = self.drum_paths[key]
                    self._cache.set(cache_key, path)
                    return path
                    
            return None
        
    def get_instrument_path(self, instrument_name: str, note: str) -> Optional[str]:
        """Get path for instrument video by note with caching"""
        with self._lock:
            self._stats['lookups'] += 1
            norm_name = instrument_name.lower().replace(' ', '_')
            note_str = str(note)
            
            # Check cache first
            cache_key = f"instrument:{norm_name}:{note_str}"
            cached_path = self._cache.get(cache_key)
            if cached_path:
                self._stats['cache_hits'] += 1
                return cached_path
            
            self._stats['cache_misses'] += 1
            
            if norm_name not in self.instrument_paths:
                return None
                
            if note_str in self.instrument_paths[norm_name]:
                path = self.instrument_paths[norm_name][note_str]
                self._cache.set(cache_key, path)
                return path
                
            # Fall back to any note for this instrument
            if self.instrument_paths[norm_name]:
                path = next(iter(self.instrument_paths[norm_name].values()))
                self._cache.set(cache_key, path)
                return path
                
            return None
    
    def batch_register_paths(self, paths_data: Dict, max_workers: int = None) -> Dict[str, bool]:
        """Register multiple paths in parallel"""
        if max_workers is None:
            max_workers = min(4, psutil.cpu_count(logical=False))
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for path_type, data in paths_data.items():
                if path_type == 'drums':
                    for drum_name, file_path in data.items():
                        future = executor.submit(self.register_drum, drum_name, file_path)
                        futures[future] = ('drum', drum_name)
                elif path_type == 'instruments':
                    for instrument_name, notes_data in data.items():
                        for note, file_path in notes_data.items():
                            future = executor.submit(self.register_instrument, instrument_name, note, file_path)
                            futures[future] = ('instrument', f"{instrument_name}:{note}")
            
            # Collect results
            for future in as_completed(futures):
                path_type, identifier = futures[future]
                try:
                    success = future.result()
                    results[f"{path_type}:{identifier}"] = success
                except Exception as e:
                    logging.error(f"Failed to register {path_type}:{identifier}: {e}")
                    results[f"{path_type}:{identifier}"] = False
        
        logging.info(f"Batch registration completed: {sum(results.values())}/{len(results)} successful")
        return results
    
    def _validate_path(self, file_path: str) -> bool:
        """Validate that a file path exists and is accessible"""
        if file_path in self._path_validation_cache:
            return self._path_validation_cache[file_path]
        
        try:
            exists = os.path.exists(file_path) and os.path.isfile(file_path)
            self._path_validation_cache[file_path] = exists
            
            # Limit cache size
            if len(self._path_validation_cache) > 1000:
                # Remove oldest entries
                keys_to_remove = list(self._path_validation_cache.keys())[:500]
                for key in keys_to_remove:
                    del self._path_validation_cache[key]
            
            return exists
        except Exception as e:
            logging.error(f"Path validation failed for {file_path}: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get registry performance statistics"""
        with self._lock:
            return {
                **self._stats.copy(),
                'cache_hit_ratio': self._stats['cache_hits'] / max(1, self._stats['lookups']),
                'total_paths': len(self.drum_paths) + sum(len(notes) for notes in self.instrument_paths.values()),
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
            }
    
    def save_registry(self, file_path: Optional[str] = None) -> bool:
        """Save registry to file for persistence between runs"""
        if file_path:
            self.registry_file = file_path
        elif not self.registry_file:
            return False
            
        data = {
            "drum_paths": self.drum_paths,
            "instrument_paths": self.instrument_paths,
            "track_paths": self.track_paths
        }
        
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
        return True
        
    def load_registry(self, file_path: Optional[str] = None) -> bool:
        """Load registry from file"""
        if file_path:
            self.registry_file = file_path
        elif not self.registry_file:
            return False
            
        if not os.path.exists(self.registry_file):
            return False
            
        with open(self.registry_file, 'r') as f:
            data = json.load(f)
            
        self.drum_paths = data.get("drum_paths", {})
        self.instrument_paths = data.get("instrument_paths", {})
        self.track_paths = data.get("track_paths", {})
        return True
        
    def register_from_directory(self, base_dir):
        """Register all videos in a directory structure"""
        base_path = Path(base_dir)
        
        # Register drum videos - using glob patterns that match your actual directory structure
        for drum_path in base_path.glob("**/drum_*.mp4"):
            try:
                # Extract drum name from filename (removing drum_ prefix)
                drum_name = drum_path.stem
                if drum_name.startswith("drum_"):
                    drum_name = drum_name[5:]  # Remove "drum_" prefix
                self.register_drum(drum_name, str(drum_path))
            except Exception as e:
                logging.error(f"Error registering drum: {e}")
                
        # Register instrument note videos
        for note_path in base_path.glob("**/note_*.mp4"):
            try:
                # Extract instrument name from parent directory
                instrument_dir = note_path.parent
                if "_notes" in instrument_dir.name:
                    instrument_name = instrument_dir.name.replace("_notes", "")
                elif "track_" in instrument_dir.parent.name:
                    instrument_name = instrument_dir.parent.name.replace("track_", "")
                else:
                    instrument_name = instrument_dir.name
                    
                # Extract note from filename
                if "note_" in note_path.stem:
                    note_parts = note_path.stem.split("_")
                    if len(note_parts) >= 2:
                        try:
                            note = note_parts[1]  # Get the numeric note value
                            self.register_instrument(instrument_name, note, str(note_path))
                        except (ValueError, IndexError):
                            logging.warning(f"Could not parse note number from {note_path}")
            except Exception as e:
                logging.error(f"Error registering instrument: {e}")

    def register_drum_path(self, drum_key, path):
        """Register a path for a drum sound by key name"""
        logging.info(f"Registered drum path: {drum_key.replace('drum_', '')} -> {path}")
        
        # Extract the drum name without 'drum_' prefix for consistent lookup
        drum_name = drum_key.replace('drum_', '')
        
        # Initialize drum_paths if it doesn't exist
        if not hasattr(self, 'drum_paths'):
            self.drum_paths = {}
            
        self.drum_paths[drum_name] = path
        return path