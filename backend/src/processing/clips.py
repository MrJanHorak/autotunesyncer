import math
from typing import List, Dict

def prepare_clips_data(video_files: Dict, duration: float) -> List[Dict]:
    """Prepare clips data with proper MIDI note mapping and positioning."""
    clips_data = []
    
    # Get total number of tracks with notes
    total_tracks = len([t for t in video_files.values() if t.get('notes')])
    
    for track_idx, (track_id, track_data) in enumerate(video_files.items()):
        if not track_data.get('notes'):
            continue

        # Calculate grid position and dimensions
        dims = calculate_track_dimensions(track_idx, total_tracks)
        
        # Process each note in the track
        for note in track_data['notes']:
            clips_data.append({
                **dims,
                'start_time': float(note['time']),
                'duration': float(note['duration']),
                'is_drum': track_data.get('isDrum', False),
                'velocity': note.get('velocity', 1.0),
                'source_path': track_data['path'],
                'midi_note': note['midi'],
                'track_id': track_id
            })

    return sorted(clips_data, key=lambda x: x['start_time'])

def calculate_track_dimensions(track_idx: int, total_tracks: int) -> Dict:
    """Calculate dimensions and position for a track in the grid."""
    if total_tracks == 1:
        return {'width': 960, 'height': 720, 'x': 0, 'y': 0}
    
    cols = math.ceil(math.sqrt(total_tracks))
    rows = math.ceil(total_tracks / cols)
    width = 960 // cols
    height = 720 // rows
    
    return {
        'width': width,
        'height': height,
        'x': (track_idx % cols) * width,
        'y': (track_idx // cols) * height
    }