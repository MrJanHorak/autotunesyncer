import os
import logging
from pathlib import Path
import json
from typing import Optional, Dict, Set
import threading
from threading import RLock
import psutil
from concurrent.futures import ThreadPoolExecutor

class LRUCache:
    """Simple LRU cache implementation"""
    def __init__(self, maxsize=128):
        self.maxsize = maxsize
        self.cache = {}
        self.access_order = []
        self.lock = RLock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
            elif len(self.cache) >= self.maxsize:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)

class PathRegistry:
    """Central registry for tracking file paths across the application"""
    
    _instance = None
    _lock = RLock()
    
    @classmethod
    def get_instance(cls):
        """Singleton access method"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = PathRegistry()
        return cls._instance
    
    def __init__(self):
        self.drum_paths = {}          # Format: {"drum_name": "path/to/file.mp4"}
        self.instrument_paths = {}    # Format: {"instrument_name": {"note_X": "path.mp4"}}
        self.track_paths = {}         # Format: {"track_id": "path/to/directory"}
        self.registry_file = None     # Path to save/load registry
        self._lock = RLock()
        self._cache = LRUCache(maxsize=256)
        self._path_validation_cache = {}
        self._stats = {
            'registrations': 0,
            'lookups': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def register_drum(self, drum_name: str, file_path: str, validate: bool = True) -> bool:
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
        
    def register_instrument(self, instrument_name: str, note: str, file_path: str, validate: bool = True) -> bool:
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
    
    def register_track_directory(self, track_id: str, directory_path: str) -> bool:
        """Register a track's base directory"""
        with self._lock:
            if not os.path.isdir(directory_path):
                logging.warning(f"Invalid track directory: {directory_path}")
                return False
            
            self.track_paths[track_id] = directory_path
            self._cache.set(f"track:{track_id}", directory_path)
            return True
        
    def get_drum_path(self, drum_key: str) -> Optional[str]:
        """Get path for drum video with improved matching and caching"""
        with self._lock:
            self._stats['lookups'] += 1
            norm_name = drum_key.lower().replace(' ', '_').replace('drum_', '')
            
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
                
            # Check for partial matches - more comprehensive matching
            for key in self.drum_paths:
                # Check if norm_name is contained in key or vice versa
                if (norm_name in key or key in norm_name or 
                    self._fuzzy_match(norm_name, key)):
                    path = self.drum_paths[key]
                    self._cache.set(cache_key, path)
                    logging.info(f"Drum path found via fuzzy match: {norm_name} -> {key} -> {path}")
                    return path
            
            logging.warning(f"No drum path found for: {drum_key} (normalized: {norm_name})")
            logging.debug(f"Available drum paths: {list(self.drum_paths.keys())}")
            return None
    
    def get_instrument_path(self, instrument_name: str, note: str) -> Optional[str]:
        """Get path for instrument video by note with improved fallback and caching"""
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
            
            # Try exact instrument name match first
            if norm_name in self.instrument_paths:
                # Try exact note match
                if note_str in self.instrument_paths[norm_name]:
                    path = self.instrument_paths[norm_name][note_str]
                    self._cache.set(cache_key, path)
                    return path
                
                # Fall back to any note for this exact instrument
                if self.instrument_paths[norm_name]:
                    path = next(iter(self.instrument_paths[norm_name].values()))
                    self._cache.set(cache_key, path)
                    logging.info(f"Instrument fallback used: {norm_name}:{note_str} -> {path}")
                    return path
            
            # Try fuzzy matching on instrument names
            for key in self.instrument_paths:
                if self._fuzzy_match(norm_name, key):
                    # Try exact note match first
                    if note_str in self.instrument_paths[key]:
                        path = self.instrument_paths[key][note_str]
                        self._cache.set(cache_key, path)
                        logging.info(f"Instrument found via fuzzy match: {norm_name} -> {key}:{note_str} -> {path}")
                        return path
                    
                    # Fall back to any note for fuzzy-matched instrument
                    if self.instrument_paths[key]:
                        path = next(iter(self.instrument_paths[key].values()))
                        self._cache.set(cache_key, path)
                        logging.info(f"Instrument fuzzy fallback: {norm_name} -> {key}:{note_str} -> {path}")
                        return path
            
            logging.warning(f"No instrument path found for: {instrument_name}:{note} (normalized: {norm_name}:{note_str})")
            logging.debug(f"Available instruments: {list(self.instrument_paths.keys())}")
            return None
    
    def _fuzzy_match(self, name1: str, name2: str) -> bool:
        """Perform fuzzy matching between two names"""
        # Simple fuzzy matching - can be enhanced with more sophisticated algorithms
        name1_parts = set(name1.replace('_', ' ').split())
        name2_parts = set(name2.replace('_', ' ').split())
        
        # Check if there's significant overlap in parts
        if name1_parts and name2_parts:
            overlap = len(name1_parts.intersection(name2_parts))
            min_parts = min(len(name1_parts), len(name2_parts))
            return overlap >= min_parts * 0.5  # At least 50% overlap
        
        return False
    
    def _validate_path(self, file_path: str) -> bool:
        """Validate that a file path exists with caching"""
        try:
            # Check cache first
            if file_path in self._path_validation_cache:
                return self._path_validation_cache[file_path]
            
            exists = os.path.exists(file_path) and os.path.isfile(file_path)
            
            # Cache the result (with size limit)
            if len(self._path_validation_cache) > 1000:
                # Clear oldest entries
                keys_to_remove = list(self._path_validation_cache.keys())[:500]
                for key in keys_to_remove:
                    del self._path_validation_cache[key]
            
            self._path_validation_cache[file_path] = exists
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
            
        try:
            data = {
                "drum_paths": self.drum_paths,
                "instrument_paths": self.instrument_paths,
                "track_paths": self.track_paths,
                "stats": self._stats
            }
            
            os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"Registry saved to: {self.registry_file}")
            return True
        except Exception as e:
            logging.error(f"Failed to save registry: {e}")
            return False
        
    def load_registry(self, file_path: Optional[str] = None) -> bool:
        """Load registry from file"""
        if file_path:
            self.registry_file = file_path
        elif not self.registry_file:
            return False
            
        if not os.path.exists(self.registry_file):
            logging.info(f"Registry file not found: {self.registry_file}")
            return False
        
        try:
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                
            self.drum_paths = data.get("drum_paths", {})
            self.instrument_paths = data.get("instrument_paths", {})
            self.track_paths = data.get("track_paths", {})
            if "stats" in data:
                self._stats.update(data["stats"])
            
            logging.info(f"Registry loaded from: {self.registry_file}")
            logging.info(f"Loaded {len(self.drum_paths)} drum paths, {sum(len(notes) for notes in self.instrument_paths.values())} instrument paths")
            return True
        except Exception as e:
            logging.error(f"Failed to load registry: {e}")
            return False
        
    def register_from_directory(self, base_dir):
        """Register all videos in a directory structure"""
        base_path = Path(base_dir)
        logging.info(f"Registering videos from directory: {base_path}")
        
        # Register drum videos - using glob patterns that match your actual directory structure
        drum_count = 0
        for drum_path in base_path.glob("**/drum_*.mp4"):
            try:
                # Extract drum name from filename (removing drum_ prefix)
                drum_name = drum_path.stem
                if drum_name.startswith("drum_"):
                    drum_name = drum_name[5:]  # Remove "drum_" prefix
                self.register_drum(drum_name, str(drum_path), validate=False)
                drum_count += 1
            except Exception as e:
                logging.error(f"Error registering drum: {e}")
                
        # Register instrument note videos
        instrument_count = 0
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
                            self.register_instrument(instrument_name, note, str(note_path), validate=False)
                            instrument_count += 1
                        except (ValueError, IndexError):
                            logging.warning(f"Could not parse note number from {note_path}")
            except Exception as e:
                logging.error(f"Error registering instrument: {e}")
        
        logging.info(f"Registered {drum_count} drum videos and {instrument_count} instrument videos from {base_path}")
    
    def register_from_uploads_directory(self, uploads_dir):
        """Register videos from the uploads directory with the specific naming pattern"""
        uploads_path = Path(uploads_dir)
        logging.info(f"Registering videos from uploads directory: {uploads_path}")
        
        drum_count = 0
        instrument_count = 0
        
        # Find all processed video files in uploads directory
        for video_file in uploads_path.glob("processed_*.mp4"):
            try:
                # Extract instrument name from filename
                # Pattern: processed_timestamp-random-instrument_name.mp4
                filename = video_file.stem  # Remove .mp4 extension
                parts = filename.split('-')
                
                if len(parts) >= 3:
                    # Join all parts after the second dash to handle names with dashes
                    instrument_name = '-'.join(parts[2:])
                    
                    if instrument_name.startswith('drum_'):
                        # Register drum video
                        drum_name = instrument_name[5:]  # Remove 'drum_' prefix
                        self.register_drum(drum_name, str(video_file), validate=False)
                        drum_count += 1
                        logging.debug(f"Registered drum: {drum_name} -> {video_file}")
                    else:
                        # Register instrument video (use a default note since we don't have note-specific videos)
                        self.register_instrument(instrument_name, "60", str(video_file), validate=False)
                        instrument_count += 1
                        logging.debug(f"Registered instrument: {instrument_name} -> {video_file}")
                else:
                    logging.warning(f"Could not parse video filename: {video_file}")
                    
            except Exception as e:
                logging.error(f"Error registering video {video_file}: {e}")
        
        logging.info(f"Registered {drum_count} drum videos and {instrument_count} instrument videos from uploads directory")
        return drum_count + instrument_count > 0
    
    def clear_cache(self):
        """Clear all caches"""
        with self._lock:
            self._cache = LRUCache(maxsize=256)
            self._path_validation_cache.clear()
            logging.info("Registry caches cleared")
    
    def debug_dump(self):
        """Debug method to dump all registered paths"""
        logging.info("=== PATH REGISTRY DEBUG DUMP ===")
        logging.info(f"Drum paths ({len(self.drum_paths)}):")
        for name, path in self.drum_paths.items():
            logging.info(f"  {name} -> {path}")
        
        logging.info(f"Instrument paths ({sum(len(notes) for notes in self.instrument_paths.values())} total):")
        for instrument, notes in self.instrument_paths.items():
            logging.info(f"  {instrument}:")
            for note, path in notes.items():
                logging.info(f"    note_{note} -> {path}")
        
        logging.info(f"Track paths ({len(self.track_paths)}):")
        for track_id, path in self.track_paths.items():
            logging.info(f"  {track_id} -> {path}")
        
        stats = self.get_stats()
        logging.info(f"Registry stats: {stats}")