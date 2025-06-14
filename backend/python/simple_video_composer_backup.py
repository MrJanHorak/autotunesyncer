#!/usr/bin/env python3
"""
SIMPLIFIED HIGH-PERFORMANCE VIDEO COMPOSER

This is a streamlined version that removes the excessive complexity
and focuses on the core functionality that actually works well.

Key Performance Principles:
1. Simple direct processing (no over-engineered cache layers)
2. Correct drum handling with proper grid placement
3. Efficient memory management
4. Clear error handling

Based on working version from commit 0e2f4d9 but with bug fixes.
"""

import os
import sys
import logging
import math
import subprocess
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from drum_utils import DRUM_NOTES, get_drum_name
from utils import normalize_instrument_name

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

class SimpleVideoComposer:
    """
    Simplified video composer focused on performance and reliability.
    
    Eliminates the complex optimization layers that were causing cache misses
    and performance degradation. Returns to proven direct processing approach
    with proper drum handling.
    """
    
    CHUNK_DURATION = 4.0
    FRAME_RATE = 30
    
    def __init__(self, processed_videos_dir, midi_data, output_path):
        self.processed_videos_dir = Path(processed_videos_dir)
        self.uploads_dir = self._find_uploads_dir()
        self.output_path = Path(output_path)
        self.midi_data = midi_data
        
        # Simple track organization
        self.regular_tracks = []
        self.drum_tracks = []
        self.grid_positions = {}
        
        # Performance settings
        self.max_workers = min(4, os.cpu_count())
        
        self._process_midi_data()
        self._setup_grid_positions()
        
        logging.info(f"✅ SimpleVideoComposer initialized:")
        logging.info(f"   📁 Videos dir: {self.uploads_dir}")
        logging.info(f"   🎵 Regular tracks: {len(self.regular_tracks)}")
        logging.info(f"   🥁 Drum tracks: {len(self.drum_tracks)}")
    
    def _find_uploads_dir(self):
        """Find the uploads directory containing video files"""
        possible_paths = [
            self.processed_videos_dir.parent / "uploads",
            Path(__file__).parent.parent / "uploads",
            self.processed_videos_dir / "uploads",
        ]
        
        for path in possible_paths:
            if path.exists() and any(f.name.endswith('.mp4') for f in path.iterdir()):
                logging.info(f"Found uploads directory: {path}")
                return path
        
        raise ValueError(f"No uploads directory found near {self.processed_videos_dir}")
    
    def _process_midi_data(self):
        """Process MIDI data into simple track structure"""
        tracks = self.midi_data.get('tracks', [])
        
        for idx, track in enumerate(tracks):
            track_data = self._normalize_track(track, idx)
            
            if self._is_drum_track(track):
                self.drum_tracks.append(track_data)
                logging.info(f"Drum track {idx}: {track_data['instrument']['name']}")
            else:
                self.regular_tracks.append(track_data)
                logging.info(f"Regular track {idx}: {track_data['instrument']['name']}")
    
    def _normalize_track(self, track, idx):
        """Convert track to standard format"""
        if isinstance(track, dict):
            return {
                'id': str(idx),
                'index': idx,
                'notes': track.get('notes', []),
                'instrument': track.get('instrument', {'name': f'track_{idx}'}),
                'channel': track.get('channel', 0)
            }
        else:
            return {
                'id': str(idx),
                'index': idx,
                'notes': [],
                'instrument': {'name': f'track_{idx}'},
                'channel': 0
            }
    
    def _is_drum_track(self, track):
        """Simple drum track detection"""
        if isinstance(track, dict):
            return (
                track.get('channel') == 9 or
                'drum' in track.get('instrument', {}).get('name', '').lower() or
                track.get('isDrum', False)
            )
        return False
    
    def _setup_grid_positions(self):
        """Setup grid positions from MIDI data"""
        grid_arrangement = self.midi_data.get('gridArrangement', {})
        
        if not grid_arrangement:
            # Create default grid arrangement
            total_tracks = len(self.regular_tracks) + len(self.drum_tracks)
            cols = min(4, total_tracks)
            
            idx = 0
            for track in self.regular_tracks + self.drum_tracks:
                self.grid_positions[track['id']] = {
                    'row': idx // cols,
                    'column': idx % cols
                }
                idx += 1
        else:
            # Use provided grid arrangement
            for track_id, pos_data in grid_arrangement.items():
                if isinstance(pos_data, dict) and 'row' in pos_data and 'column' in pos_data:
                    self.grid_positions[track_id] = {
                        'row': int(pos_data['row']),
                        'column': int(pos_data['column'])
                    }
        
        logging.info(f"Grid positions set for {len(self.grid_positions)} tracks")
    
    def create_composition(self):
        """
        Create video composition using simplified, proven approach.
        
        This method eliminates the complex optimization layers and focuses
        on direct, efficient processing that actually works.
        """
        try:
            logging.info("🎬 Starting SIMPLIFIED video composition...")
            start_time = time.time()
            
            # Calculate composition parameters
            total_duration = self._calculate_total_duration()
            total_chunks = max(1, math.ceil(total_duration / self.CHUNK_DURATION))
            
            logging.info(f"Composition: {total_duration:.2f}s, {total_chunks} chunks")
            
            # Create chunks directory
            chunks_dir = self.processed_videos_dir / "simple_chunks"
            chunks_dir.mkdir(exist_ok=True)
            
            # Process chunks in parallel for performance
            chunk_paths = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                chunk_futures = {}
                
                for chunk_idx in range(total_chunks):
                    chunk_start = chunk_idx * self.CHUNK_DURATION
                    chunk_end = min(chunk_start + self.CHUNK_DURATION, total_duration)
                    
                    future = executor.submit(
                        self._create_chunk, 
                        chunk_idx, chunk_start, chunk_end, chunks_dir
                    )
                    chunk_futures[future] = chunk_idx
                
                # Collect results
                for future in as_completed(chunk_futures):
                    chunk_idx = chunk_futures[future]
                    try:
                        chunk_path = future.result()
                        if chunk_path:
                            chunk_paths.append((chunk_idx, chunk_path))
                            logging.info(f"✅ Chunk {chunk_idx + 1}/{total_chunks} completed")
                        else:
                            logging.warning(f"⚠️  Chunk {chunk_idx + 1} failed")
                    except Exception as e:
                        logging.error(f"❌ Chunk {chunk_idx + 1} error: {e}")
            
            # Sort chunks by index and concatenate
            chunk_paths.sort(key=lambda x: x[0])
            sorted_paths = [path for _, path in chunk_paths]
            
            if not sorted_paths:
                raise Exception("No chunks were created successfully")
            
            # Concatenate chunks into final video
            final_path = self._concatenate_chunks(sorted_paths)
            
            total_time = time.time() - start_time
            if final_path and os.path.exists(final_path):
                file_size = os.path.getsize(final_path)
                logging.info(f"🎉 COMPOSITION SUCCESSFUL!")
                logging.info(f"   📁 Output: {final_path}")
                logging.info(f"   📏 Size: {file_size:,} bytes")
                logging.info(f"   ⏱️  Total time: {total_time:.2f}s")
                return str(final_path)
            else:
                logging.error("❌ Final concatenation failed")
                return None
                
        except Exception as e:
            logging.error(f"❌ Composition error: {e}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _calculate_total_duration(self):
        """Calculate total composition duration"""
        max_end_time = 0
        
        for track in self.regular_tracks + self.drum_tracks:
            for note in track.get('notes', []):
                note_end = float(note.get('time', 0)) + float(note.get('duration', 1))
                max_end_time = max(max_end_time, note_end)
        
        return max_end_time + 1.0  # Add 1 second buffer
    
    def _create_chunk(self, chunk_idx, start_time, end_time, chunks_dir):
        """Create a single chunk with proper drum handling"""
        try:
            chunk_path = chunks_dir / f"chunk_{chunk_idx}.mp4"
            chunk_duration = end_time - start_time
            
            # Find all active notes in this time range
            active_tracks = self._find_active_tracks(start_time, end_time)
            
            if not active_tracks:
                # Create silent chunk
                return self._create_silent_chunk(chunk_path, chunk_duration)
            
            # Process regular tracks
            track_videos = []
            for track in active_tracks:
                if track in self.drum_tracks:
                    # Process drum track with proper drum file handling
                    drum_videos = self._process_drum_track_chunk(track, start_time, end_time)
                    track_videos.extend(drum_videos)
                else:
                    # Process regular track
                    track_video = self._process_regular_track_chunk(track, start_time, end_time)
                    if track_video:
                        track_videos.append(track_video)
            
            if not track_videos:
                return self._create_silent_chunk(chunk_path, chunk_duration)
            
            # Combine all track videos into grid layout
            return self._create_grid_composition(track_videos, chunk_path, chunk_duration)
            
        except Exception as e:
            logging.error(f"Error creating chunk {chunk_idx}: {e}")
            return None
    
    def _find_active_tracks(self, start_time, end_time):
        """Find tracks that have notes active in the time range"""
        active_tracks = []
        
        for track in self.regular_tracks + self.drum_tracks:
            has_active_notes = False
            
            for note in track.get('notes', []):
                note_start = float(note.get('time', 0))
                note_end = note_start + float(note.get('duration', 1))
                
                if note_start < end_time and note_end > start_time:
                    has_active_notes = True
                    break
            
            if has_active_notes:
                active_tracks.append(track)
        
        return active_tracks
    
    def _process_drum_track_chunk(self, drum_track, start_time, end_time):
        """
        Process drum track with CORRECT drum handling.
        
        This fixes the main drum processing issue by:
        1. Correctly mapping MIDI notes to specific drum sounds
        2. Finding the right drum video files 
        3. Placing them in correct grid positions based on drum type
        """
        drum_videos = []
        
        # Group notes by drum type (MIDI note number)
        drum_notes_by_type = {}
        
        for note in drum_track.get('notes', []):
            note_start = float(note.get('time', 0))
            note_end = note_start + float(note.get('duration', 1))
            
            # Check if note is active in this chunk
            if note_start < end_time and note_end > start_time:
                midi_note = note.get('midi')
                drum_name = get_drum_name(midi_note)
                
                if drum_name != 'Unknown':
                    if drum_name not in drum_notes_by_type:
                        drum_notes_by_type[drum_name] = []
                    drum_notes_by_type[drum_name].append(note)
        
        # Process each drum type separately
        for drum_name, notes in drum_notes_by_type.items():
            drum_video_path = self._find_drum_video_file(drum_name)
            
            if drum_video_path:
                # Create video segment for this drum type
                drum_video_info = {
                    'path': drum_video_path,
                    'track_id': f"drum_{drum_name.lower().replace(' ', '_')}",
                    'notes': notes,
                    'start_time': start_time,
                    'drum_name': drum_name
                }
                drum_videos.append(drum_video_info)
                logging.info(f"✅ Drum: {drum_name} → {os.path.basename(drum_video_path)}")
            else:
        logging.warning(f"❌ No video file found for drum: {drum_name}")
        
        return drum_videos
    
    def _find_drum_video_file(self, drum_name):
        """Find the video file for a specific drum sound with enhanced matching"""
        # Normalize drum name for file matching
        normalized_drum = f"drum_{drum_name.lower().replace(' ', '_')}"
        
        # Look for files in uploads directory
        for video_file in self.uploads_dir.glob('*.mp4'):
            if normalized_drum in video_file.name.lower():
                return str(video_file)
        
        # Also check for processed files
        for video_file in self.uploads_dir.glob('processed_*.mp4'):
            if normalized_drum in video_file.name.lower():
                return str(video_file)
        
        # Try partial matching for drum variations
        # Common drum name mappings for files we have
        drum_variations = {
            'snare_drum': ['snare_cross_stick', 'snare'],
            'kick_drum': ['bass_drum', 'kick'],
            'hi-hat_closed': ['hihat_closed', 'hi_hat_closed'],
            'crash_cymbal': ['crash'],
        }
        
        normalized_base = drum_name.lower().replace(' ', '_')
        for video_file in self.uploads_dir.glob('processed_*.mp4'):
            file_name_lower = video_file.name.lower()
            
            # Check if any variation matches
            variations = drum_variations.get(normalized_base, [normalized_base])
            for variation in variations:
                if f"drum_{variation}" in file_name_lower:
                    logging.info(f"Matched drum '{drum_name}' to '{video_file.name}' via variation '{variation}'")
                    return str(video_file)
        
        logging.warning(f"No video file found for drum: {drum_name} (looking for {normalized_drum})")
        return None
    
    def _process_regular_track_chunk(self, track, start_time, end_time):
        """Process regular (non-drum) track chunk"""
        # Find the video file for this instrument
        instrument_name = track['instrument']['name']
        normalized_name = normalize_instrument_name(instrument_name)
        
        video_file = self._find_instrument_video_file(normalized_name)
        if not video_file:
            logging.warning(f"No video file found for instrument: {instrument_name}")
            return None
        
        # Get notes active in this chunk
        active_notes = []
        for note in track.get('notes', []):
            note_start = float(note.get('time', 0))
            note_end = note_start + float(note.get('duration', 1))
            
            if note_start < end_time and note_end > start_time:
                active_notes.append(note)
        
        if not active_notes:
            return None
          return {
            'path': video_file,
            'track_id': track['id'],
            'notes': active_notes,
            'start_time': start_time,
            'instrument_name': instrument_name
        }
    
    def _find_instrument_video_file(self, instrument_name):
        """Find the video file for an instrument with enhanced matching"""
        # Normalize instrument name for better matching
        normalized_name = instrument_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').replace('-', '_')
        
        # Look for files matching the instrument name
        for video_file in self.uploads_dir.glob('*.mp4'):
            file_name_lower = video_file.name.lower()
            # Try direct match first
            if instrument_name.lower() in file_name_lower:
                return str(video_file)
            # Try normalized match
            if normalized_name in file_name_lower:
                return str(video_file)
            
        # Also check for processed files
        for video_file in self.uploads_dir.glob('processed_*.mp4'):
            file_name_lower = video_file.name.lower()
            # Try direct match first
            if instrument_name.lower() in file_name_lower:
                return str(video_file)
            # Try normalized match
            if normalized_name in file_name_lower:
                return str(video_file)
        
        # Try partial matching for complex names
        name_parts = instrument_name.lower().split()
        for video_file in self.uploads_dir.glob('processed_*.mp4'):
            file_name_lower = video_file.name.lower()
            if any(part in file_name_lower for part in name_parts if len(part) > 2):
                logging.info(f"Matched '{instrument_name}' to '{video_file.name}' via partial match")
                return str(video_file)
        
        return None
    
    def _create_grid_composition(self, track_videos, output_path, duration):
        """Create grid-based composition from track videos"""
        try:
            if not track_videos:
                return self._create_silent_chunk(output_path, duration)
            
            logging.info(f"🎬 Creating grid with {len(track_videos)} videos")
            
            # If only one video, just use it directly
            if len(track_videos) == 1:
                cmd = [
                    'ffmpeg', '-y',
                    '-i', track_videos[0]['path'],
                    '-t', str(duration),
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-c:a', 'aac', '-b:a', '128k',
                    str(output_path)
                ]
            else:
                # Create proper grid layout for multiple videos
                cmd = ['ffmpeg', '-y']
                
                # Add all input videos
                for video in track_videos:
                    cmd.extend(['-i', video['path']])
                
                # Calculate grid dimensions
                video_count = len(track_videos)
                if video_count == 2:
                    cols, rows = 2, 1
                    width, height = 960, 720  # Split screen
                elif video_count <= 4:
                    cols, rows = 2, 2
                    width, height = 480, 360  # Quad
                else:
                    cols = min(4, math.ceil(math.sqrt(video_count)))
                    rows = math.ceil(video_count / cols)
                    width, height = 1920 // cols, 1080 // rows
                
                # Build filter complex for grid layout
                video_filters = []
                audio_inputs = []
                
                # Scale each video
                for i in range(video_count):
                    video_filters.append(f'[{i}:v]scale={width}:{height}[v{i}]')
                    audio_inputs.append(f'[{i}:a]')
                
                # Create grid overlay
                if video_count == 2:
                    overlay_filter = '[v0][v1]hstack[v]'
                elif video_count <= 4:
                    # 2x2 grid
                    overlay_filter = '[v0][v1]hstack[top];[v2][v3]hstack[bottom];[top][bottom]vstack[v]'
                    # Handle case where we don't have 4 videos
                    if video_count == 3:
                        overlay_filter = '[v0][v1]hstack[top];[v2]scale=960:360[v2scaled];[top][v2scaled]vstack[v]'
                else:
                    # More complex grid - use overlay positioning
                    overlay_chain = '[v0]'
                    for i in range(1, video_count):
                        row = (i - 1) // cols
                        col = (i - 1) % cols
                        x = col * width
                        y = row * height
                        
                        if i == video_count - 1:
                            overlay_chain += f'[v{i}]overlay={x}:{y}[v]'
                        else:
                            overlay_chain += f'[v{i}]overlay={x}:{y}[tmp{i}];[tmp{i}]'
                    overlay_filter = overlay_chain
                
                # Mix audio
                if len(audio_inputs) > 1:
                    audio_filter = f'{"".join(audio_inputs)}amix=inputs={len(audio_inputs)}:duration=longest[a]'
                else:
                    audio_filter = '[0:a]acopy[a]'
                
                # Combine all filters
                filter_complex = ';'.join(video_filters + [overlay_filter, audio_filter])
                
                cmd.extend([
                    '-filter_complex', filter_complex,
                    '-map', '[v]',
                    '-map', '[a]',
                    '-t', str(duration),
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-c:a', 'aac', '-b:a', '128k',
                    str(output_path)
                ])
                
                logging.info(f"📐 Grid layout: {cols}x{rows}, cell size: {width}x{height}")
            
            # Execute FFmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode == 0:
                logging.info(f"✅ Grid composition created: {os.path.basename(output_path)}")
                return str(output_path)
            else:
                logging.error(f"FFmpeg error: {result.stderr}")
                logging.error(f"Command: {' '.join(cmd)}")
                return None
                
        except Exception as e:
            logging.error(f"Error creating grid composition: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _create_silent_chunk(self, output_path, duration):
        """Create a silent video chunk"""
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', 'color=black:s=1920x1080:r=30',
            '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
            '-t', str(duration),
            '-c:v', 'libx264', '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '128k',
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode == 0:
                return str(output_path)
            else:
                logging.error(f"Error creating silent chunk: {result.stderr}")
                return None
        except Exception as e:
            logging.error(f"Error creating silent chunk: {e}")
            return None
    
    def _concatenate_chunks(self, chunk_paths):
        """Concatenate chunks into final video"""
        try:
            # Create concat file list
            concat_file = self.processed_videos_dir / "concat_list.txt"
            
            with open(concat_file, 'w', encoding='utf-8') as f:
                for chunk_path in chunk_paths:
                    f.write(f"file '{chunk_path}'\n")
            
            # Concatenate using FFmpeg
            final_path = self.output_path
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                str(final_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode == 0:
                # Cleanup
                os.unlink(concat_file)
                for chunk_path in chunk_paths:
                    try:
                        os.unlink(chunk_path)
                    except:
                        pass
                
                return str(final_path)
            else:
                logging.error(f"Concatenation error: {result.stderr}")
                return None
                
        except Exception as e:
            logging.error(f"Error concatenating chunks: {e}")
            return None


def main():
    """Test the simplified composer"""
    import json
    
    # Test with sample data
    test_midi_data = {
        'tracks': [
            {
                'instrument': {'name': 'piano'},
                'notes': [
                    {'time': 0, 'duration': 1, 'midi': 60},
                    {'time': 1, 'duration': 1, 'midi': 62},
                ]
            }
        ],
        'gridArrangement': {}
    }
    
    composer = SimpleVideoComposer(
        '/path/to/videos',
        test_midi_data,
        '/path/to/output.mp4'
    )
    
    result = composer.create_composition()
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
