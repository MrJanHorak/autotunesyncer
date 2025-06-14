#!/usr/bin/env python3
"""
VideoComposer Wrapper
Provides a clean interface to use the efficient chunk-based VideoComposer system
instead of the problematic note-by-note video_processor.py
"""

import sys
import os
import logging
from pathlib import Path

# Add the python directory to path to import VideoComposer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

from video_composer import VideoComposer
import tempfile
import json

class VideoComposerWrapper:
    """
    Wrapper around the efficient VideoComposer system to replace video_processor.py
    
    This provides the same interface as video_processor.py but uses the much more
    efficient chunk-based processing approach instead of processing 136 individual
    notes during final combination.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.temp_dir = None
        self.composer = None
        
    def _transform_midi_data(self, midi_data: dict) -> dict:
        """
        Transform MIDI data format to match VideoComposer expectations
        Convert 'start'/'end' fields to 'time' fields and ensure proper structure
        """
        try:
            self.logger.info("Transforming MIDI data format for VideoComposer compatibility")
            
            # Create a copy to avoid modifying original data
            transformed_data = json.loads(json.dumps(midi_data))
            
            # Ensure tracks is a list
            if 'tracks' in transformed_data:
                tracks = transformed_data['tracks']
                
                # If tracks is a dict, convert to list
                if isinstance(tracks, dict):
                    track_list = []
                    for track_id, track_data in tracks.items():
                        if isinstance(track_data, dict):
                            track_data['id'] = track_id
                            track_list.append(track_data)
                    transformed_data['tracks'] = track_list
                
                # Transform note format for each track
                for track in transformed_data['tracks']:
                    if 'notes' in track and isinstance(track['notes'], list):
                        for note in track['notes']:
                            # Convert start/end to time field if needed
                            if 'start' in note and 'time' not in note:
                                note['time'] = note['start']
                            if 'end' in note and 'duration' not in note:
                                note['duration'] = note['end'] - note.get('start', 0)
                            
                            # Ensure required fields exist
                            if 'time' not in note:
                                note['time'] = 0.0
                            if 'duration' not in note:
                                note['duration'] = 1.0
                            if 'pitch' not in note:
                                note['pitch'] = 60  # Middle C default
            
            self.logger.info(f"Transformed {len(transformed_data.get('tracks', []))} tracks for VideoComposer")
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Failed to transform MIDI data: {e}")
            return midi_data  # Return original if transformation fails

    def process_videos(self, midi_data: dict, video_files: dict, output_path: str) -> bool:
        """
        Main entry point that replaces video_processor.py functionality
        
        Args:
            midi_data: MIDI data with tracks and notes
            video_files: Dictionary of video files (kept for compatibility)
            output_path: Final output video path
              Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info("Starting VideoComposer-based processing (efficient chunk-based approach)")
            
            # Transform MIDI data format to match VideoComposer expectations
            midi_data = self._transform_midi_data(midi_data)
            
            # Create temporary directory with proper structure for VideoComposer  
            self.temp_dir = Path(tempfile.mkdtemp(prefix='video_composer_'))
            self.logger.info(f"Using temp directory: {self.temp_dir}")
            
            # Create the directory structure that VideoComposer expects
            # VideoComposer looks for uploads at: processed_videos_dir/../../../uploads
            # So if processed_videos_dir is: temp_dir/a/b/processed
            # Then uploads should be at: temp_dir/uploads
            uploads_dir = self.temp_dir / "uploads"
            processed_dir = self.temp_dir / "a" / "b" / "processed"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            processed_dir.mkdir(parents=True, exist_ok=True)
              # Ensure grid arrangement is present before initializing VideoComposer
            if 'gridArrangement' not in midi_data or not midi_data['gridArrangement']:
                self.logger.error("No grid arrangement found in MIDI data")
                raise ValueError("Grid arrangement is required for video composition")
            
            self.logger.info(f"Grid arrangement validated: {midi_data['gridArrangement']}")
            
            # Initialize VideoComposer with the correct path structure
            # The VideoComposer will access gridArrangement from midi_data during __init__
            self.composer = VideoComposer(
                processed_videos_dir=str(processed_dir),
                midi_data=midi_data,
                output_path=output_path
            )
            
            self.logger.info("VideoComposer initialized successfully")
            
            # Use the efficient chunk-based composition
            result_path = self.composer.create_composition()
            
            if result_path and os.path.exists(result_path):
                self.logger.info(f"VideoComposer completed successfully: {result_path}")
                return True
            else:
                self.logger.error("VideoComposer failed to create composition")
                return False
                
        except Exception as e:
            self.logger.error(f"VideoComposer processing failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
        finally:
            self.cleanup()
    
    def process_video_with_notes(self, midi_data: dict, video_files: dict, output_path: str) -> bool:
        """
        Alternative method name for compatibility - delegates to process_videos
        """
        return self.process_videos(midi_data, video_files, output_path)
    
    def combine_videos(self, processed_videos: dict, midi_data: dict, output_path: str) -> bool:
        """
        Compatibility method for video combination
        This method is mainly for interface compatibility since VideoComposer
        handles combination internally as part of its chunk-based approach.
        """
        try:
            if not processed_videos:
                self.logger.warning("No processed videos provided to combine_videos")
                return False
                
            # If we only have one video, just copy it
            if len(processed_videos) == 1:
                import shutil
                video_path = list(processed_videos.values())[0]
                shutil.copy2(video_path, output_path)
                self.logger.info(f"Single video copied to output: {output_path}")
                return True
            
            # For multiple videos, we would typically use VideoComposer's combination
            # but since VideoComposer handles this internally, we'll just use the first video
            self.logger.warning("combine_videos called with multiple videos - VideoComposer handles combination internally")
            return False
            
        except Exception as e:
            self.logger.error(f"Error in combine_videos: {e}")
            return False

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and self.temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                self.logger.info("Temporary files cleaned up")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp directory: {e}")
    
    def report_progress(self, percentage: int, message: str):
        """Report progress (compatibility with video_processor interface)"""
        self.logger.info(f"Progress {percentage}%: {message}")


def main():
    """
    Main function for testing the wrapper
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='VideoComposer Wrapper Test')
    parser.add_argument('--midi', required=True, help='Path to MIDI JSON file')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load MIDI data
    try:
        with open(args.midi, 'r') as f:
            midi_data = json.load(f)
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return 1
    
    # Create wrapper and process
    wrapper = VideoComposerWrapper()
    success = wrapper.process_videos(
        midi_data=midi_data,
        video_files={},  # Empty for compatibility
        output_path=args.output
    )
    
    if success:
        print(f"Video composition completed successfully: {args.output}")
        return 0
    else:
        print("Video composition failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
