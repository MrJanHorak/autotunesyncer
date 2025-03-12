import os
import logging
from pathlib import Path
import json

class PathRegistry:
    """Central registry for tracking file paths across the application"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Singleton access method"""
        if cls._instance is None:
            cls._instance = PathRegistry()
        return cls._instance
    
    def __init__(self):
        self.drum_paths = {}          # Format: {"drum_name": "path/to/file.mp4"}
        self.instrument_paths = {}    # Format: {"instrument_name": {"note_X": "path.mp4"}}
        self.track_paths = {}         # Format: {"track_id": "path/to/directory"}
        self.debug_lookups = {}       # For tracking lookup attempts
        self.registry_file = None     # Path to save/load registry
    
    def register_drum(self, drum_name, file_path):
        """Register a drum video path"""
        norm_name = drum_name.lower().replace(' ', '_')
        self.drum_paths[norm_name] = file_path
        logging.info(f"Registered drum path: {norm_name} -> {file_path}")
        
    def register_instrument(self, instrument_name, note, file_path):
        """Register an instrument note video path"""
        norm_name = instrument_name.lower().replace(' ', '_')
        if norm_name not in self.instrument_paths:
            self.instrument_paths[norm_name] = {}
        self.instrument_paths[norm_name][str(note)] = file_path
        logging.info(f"Registered instrument path: {norm_name}:{note} -> {file_path}")
    
    def register_track_directory(self, track_id, directory_path):
        """Register a track's base directory"""
        self.track_paths[track_id] = directory_path
        
    def get_drum_path(self, drum_key):
        """Get path for drum video"""
        norm_name = drum_key.lower().replace(' ', '_')
        
        # Direct lookup
        if norm_name in self.drum_paths:
            return self.drum_paths[norm_name]
            
        # Check for partial matches
        for key in self.drum_paths:
            if norm_name in key or key in norm_name:
                return self.drum_paths[key]
                
        return None
        
    def get_instrument_path(self, instrument_name, note):
        """Get path for instrument video by note"""
        norm_name = instrument_name.lower().replace(' ', '_')
        note_str = str(note)
        
        if norm_name not in self.instrument_paths:
            return None
            
        if note_str in self.instrument_paths[norm_name]:
            return self.instrument_paths[norm_name][note_str]
            
        # Fall back to any note for this instrument
        if self.instrument_paths[norm_name] and len(self.instrument_paths[norm_name]) > 0:
            return next(iter(self.instrument_paths[norm_name].values()))
            
        return None
        
    def save_registry(self, file_path=None):
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
        
    def load_registry(self, file_path=None):
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