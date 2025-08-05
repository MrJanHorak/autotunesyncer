#!/usr/bin/env python3
"""
GPU-accelerated note synchronization for video processing
Integrates with the main video processing pipeline to provide fast note-synchronized video composition
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'config'))

from ffmpeg_gpu import ffmpeg_gpu_encode, gpu_batch_process, gpu_note_synchronized_encode
from video_composer_wrapper import VideoComposerWrapper
from gpu_config import FFMPEG_GPU_CONFIG

class GPUNoteSynchronizer:
    """
    High-performance GPU-accelerated note synchronization for video processing
    Integrates with the existing video processing pipeline
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.video_composer = VideoComposerWrapper()
        
    def process_with_gpu_acceleration(self, midi_data: Dict, video_files: Dict, output_path: str) -> bool:
        """
        Process videos with GPU acceleration for note synchronization
        
        Args:
            midi_data: MIDI data with tracks and notes
            video_files: Dictionary of video files  
            output_path: Final output video path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info("Starting GPU-accelerated note synchronization")
            
            # Add GPU memory management
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU cache before processing
                self.logger.info("GPU memory cleared before processing")
            
            # Get note timing data for GPU processing
            note_timings = self._extract_note_timings(midi_data)
            
            # Use GPU note-synchronized encoding
            if note_timings:
                self.logger.info(f"Processing {len(note_timings)} note events with GPU acceleration")
                
                # Create output directory for GPU processing
                import tempfile
                import os
                gpu_output_dir = tempfile.mkdtemp(prefix='gpu_processing_')
                
                # Create GPU-optimized encoding commands
                gpu_commands = self._create_gpu_commands(note_timings, video_files, output_path)
                
                # Execute GPU batch processing with proper parameters
                # Convert commands to input_files format expected by gpu_batch_process
                input_files = []
                for cmd in gpu_commands:
                    if '-i' in cmd:
                        i_index = cmd.index('-i')
                        if i_index + 1 < len(cmd):
                            input_files.append(cmd[i_index + 1])
                
                if input_files:
                    results = gpu_batch_process(input_files, gpu_output_dir)
                    
                    # Check if batch processing succeeded
                    success = results and len(results) > 0
                    
                    if success:
                        self.logger.info("GPU-accelerated note synchronization completed successfully")
                        # Clean up temporary directory
                        import shutil
                        shutil.rmtree(gpu_output_dir, ignore_errors=True)
                        
                        # Clear GPU memory after processing
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        return True
                    else:
                        self.logger.warning("GPU batch processing failed, falling back to standard processing")
                        # Clean up temporary directory
                        import shutil
                        shutil.rmtree(gpu_output_dir, ignore_errors=True)
                else:
                    self.logger.warning("No valid input files found for GPU processing")
                    # Clean up temporary directory
                    import shutil
                    shutil.rmtree(gpu_output_dir, ignore_errors=True)
            
            # Clear GPU memory before fallback
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Fallback to standard processing if GPU processing fails
            return self.video_composer.process_videos(midi_data, video_files, output_path)
            
        except Exception as e:
            self.logger.error(f"GPU note synchronization failed: {e}")
            
            # Emergency GPU memory cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info("Emergency GPU memory cleanup completed")
            except:
                pass
            
            # Fallback to standard processing
            return self.video_composer.process_videos(midi_data, video_files, output_path)
    
    def _extract_note_timings(self, midi_data: Dict) -> List[Dict]:
        """Extract note timing data for GPU processing"""
        note_timings = []
        
        try:
            tracks = midi_data.get('tracks', [])
            
            for track in tracks:
                if isinstance(track, dict) and 'notes' in track:
                    track_id = track.get('id', 'unknown')
                    
                    for note in track['notes']:
                        timing = {
                            'track_id': track_id,
                            'time': note.get('time', note.get('start', 0)),
                            'duration': note.get('duration', note.get('end', 1) - note.get('start', 0)),
                            'pitch': note.get('pitch', 60),
                            'velocity': note.get('velocity', 100)
                        }
                        note_timings.append(timing)
            
            # Sort by time for optimal GPU processing
            note_timings.sort(key=lambda x: x['time'])
            
            self.logger.info(f"Extracted {len(note_timings)} note timings for GPU processing")
            return note_timings
            
        except Exception as e:
            self.logger.error(f"Failed to extract note timings: {e}")
            return []
    
    def _create_gpu_commands(self, note_timings: List[Dict], video_files: Dict, output_path: str) -> List[List[str]]:
        """Create GPU-optimized FFmpeg commands for note synchronization"""
        commands = []
        
        try:
            # Create commands for note-synchronized video segments
            for i, timing in enumerate(note_timings):
                segment_output = f"{output_path}_segment_{i}.mp4"
                
                # Build GPU-accelerated command
                cmd = [
                    'ffmpeg', '-y',
                    '-hwaccel', 'cuda',
                    '-i', str(video_files.get(timing['track_id'], '')),
                    '-ss', str(timing['time']),
                    '-t', str(timing['duration']),
                    '-c:v', FFMPEG_GPU_CONFIG['encoder'],
                    '-preset', FFMPEG_GPU_CONFIG['preset'],
                    '-b:v', FFMPEG_GPU_CONFIG['bitrate'],
                    '-maxrate', FFMPEG_GPU_CONFIG['max_bitrate'],
                    '-bufsize', FFMPEG_GPU_CONFIG['buffer_size'],
                    '-pix_fmt', FFMPEG_GPU_CONFIG['pixel_format'],
                    segment_output
                ]
                
                commands.append(cmd)
            
            # Add final composition command
            if commands:
                composition_cmd = self._create_composition_command(note_timings, output_path)
                commands.append(composition_cmd)
            
            self.logger.info(f"Created {len(commands)} GPU commands for note synchronization")
            return commands
            
        except Exception as e:
            self.logger.error(f"Failed to create GPU commands: {e}")
            return []
    
    def _create_composition_command(self, note_timings: List[Dict], output_path: str) -> List[str]:
        """Create final composition command with GPU acceleration"""
        
        # Create filter_complex for combining segments
        filter_parts = []
        input_args = []
        
        for i, timing in enumerate(note_timings):
            segment_path = f"{output_path}_segment_{i}.mp4"
            input_args.extend(['-i', segment_path])
            filter_parts.append(f"[{i}:v]")
        
        # Simple concatenation for now - can be enhanced with grid layout
        filter_complex = f"{''.join(filter_parts)}concat=n={len(note_timings)}:v=1:a=0[out]"
        
        cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'cuda'
        ]
        
        cmd.extend(input_args)
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-c:v', FFMPEG_GPU_CONFIG['encoder'],
            '-preset', FFMPEG_GPU_CONFIG['preset'],
            '-b:v', FFMPEG_GPU_CONFIG['bitrate'],
            output_path
        ])
        
        return cmd

# Integration function for existing video processing pipeline
def integrate_gpu_acceleration(midi_data: Dict, video_files: Dict, output_path: str) -> bool:
    """
    Main integration function that can be called from existing video processing pipeline
    
    This function provides a drop-in replacement for CPU-based video processing
    with GPU acceleration for note synchronization
    """
    try:
        gpu_synchronizer = GPUNoteSynchronizer()
        return gpu_synchronizer.process_with_gpu_acceleration(midi_data, video_files, output_path)
    except Exception as e:
        logging.error(f"GPU integration failed: {e}")
        return False

if __name__ == "__main__":
    # Test GPU note synchronization
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    sample_midi = {
        'tracks': [
            {
                'id': 'track1',
                'notes': [
                    {'time': 0.0, 'duration': 1.0, 'pitch': 60},
                    {'time': 1.0, 'duration': 1.0, 'pitch': 64}
                ]
            }
        ]
    }
    
    sample_videos = {
        'track1': 'test_video.mp4'
    }
    
    gpu_sync = GPUNoteSynchronizer()
    result = gpu_sync.process_with_gpu_acceleration(sample_midi, sample_videos, 'test_output.mp4')
    print(f"GPU note synchronization test: {'SUCCESS' if result else 'FAILED'}")
