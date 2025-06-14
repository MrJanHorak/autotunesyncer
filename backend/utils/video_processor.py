#!/usr/bin/env python3
"""
Drop-in replacement for the problematic video_processor.py
Uses the efficient chunk-based VideoComposer architecture instead of note-by-note processing
"""

import sys
import os
import logging
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.video_composer_wrapper import VideoComposerWrapper

# Configure logging without Unicode characters to prevent Windows CP1252 encoding errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Configure stream handler to handle Unicode properly on Windows
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        # Ensure proper encoding for Windows console
        if hasattr(handler.stream, 'reconfigure'):
            try:
                handler.stream.reconfigure(encoding='utf-8', errors='replace')
            except:
                pass

logger = logging.getLogger(__name__)

class EnhancedVideoProcessor:
    """
    Drop-in replacement that uses VideoComposerWrapper internally
    Maintains the same interface as the original but uses chunk-based processing
    """
    
    def __init__(self, performance_mode=True, memory_limit_gb=4, parallel_tracks=None):
        """Initialize with VideoComposerWrapper backend"""
        self.performance_mode = performance_mode
        self.memory_limit_gb = memory_limit_gb
        self.parallel_tracks = parallel_tracks
        
        # Use the efficient VideoComposerWrapper internally
        self.composer_wrapper = VideoComposerWrapper()
        
        logger.info(f"Initialized EnhancedVideoProcessor with chunk-based backend")
        logger.info(f"Performance mode: {performance_mode}, Memory limit: {memory_limit_gb}GB")
    
    def report_progress(self, progress: int, message: str = ""):
        """Report progress to parent process"""
        print(f"PROGRESS:{progress}", flush=True)
        if message:
            logger.info(f"Progress {progress}%: {message}")
        
        # Also delegate to wrapper
        self.composer_wrapper.report_progress(progress, message)
    
    def validate_input_files(self, midi_json_path: str, video_files_json_path: str) -> bool:
        """Validate that input files exist and are readable"""
        try:
            if not os.path.exists(midi_json_path):
                logger.error(f"MIDI JSON file not found: {midi_json_path}")
                return False
                
            if not os.path.exists(video_files_json_path):
                logger.error(f"Video files JSON not found: {video_files_json_path}")
                return False
                  # Try to load JSON files
            with open(midi_json_path, 'r') as f:
                json.load(f)
            with open(video_files_json_path, 'r') as f:
                json.load(f)
                
            return True
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def process_videos(self, midi_data: Dict, video_files: Dict, output_path: str) -> bool:
        """
        Main processing method - delegates to chunk-based VideoComposerWrapper
        This replaces the problematic note-by-note processing with efficient chunking
        """
        try:
            logger.info("Using efficient chunk-based processing instead of note-by-note")
            logger.info(f"Processing {len(midi_data.get('tracks', []))} tracks with chunked approach")
            
            # Add grid arrangement validation
            grid_arrangement = midi_data.get('gridArrangement', {})
            logger.info(f"Grid arrangement validation: {len(grid_arrangement)} positions found")
            if not grid_arrangement:
                logger.error("ERROR: Grid arrangement is empty in video processor")
                return False
            
            # Count total notes for comparison
            total_notes = 0
            for track in midi_data.get('tracks', []):
                if isinstance(track, dict) and 'notes' in track:
                    total_notes += len(track.get('notes', []))
            
            logger.info(f"Total notes to process: {total_notes}")
            logger.info("Instead of processing each note individually, using 4-second chunks")
            
            # Delegate to the efficient VideoComposerWrapper
            success = self.composer_wrapper.process_videos(midi_data, video_files, output_path)
            
            if success:
                logger.info("SUCCESS: Chunk-based processing completed successfully!")
                return True
            else:
                logger.error("ERROR: Chunk-based processing failed")
                return False
                
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def combine_videos(self, processed_videos: Dict[str, str], midi_data: Dict, output_path: str) -> bool:
        """
        Combine videos using chunk-based approach instead of complex FFmpeg filters
        """
        logger.info("Using chunk-based combination instead of complex FFmpeg filters")
        return self.composer_wrapper.combine_videos(processed_videos, midi_data, output_path)
    
    def process_video_with_notes(self, midi_data: Dict, video_files: Dict, output_path: str) -> bool:
        """Alternative interface for compatibility"""
        return self.process_videos(midi_data, video_files, output_path)
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            self.composer_wrapper.cleanup()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    def cleanup(self):
        """Alias for cleanup_temp_files"""
        self.cleanup_temp_files()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Return performance summary of the chunk-based processing"""
        return {
            'processing_method': 'chunk-based',
            'architecture': 'VideoComposer with GPU acceleration',
            'chunking': '4-second segments',
            'parallel_processing': True,
            'gpu_acceleration': True,
            'note_handling': 'pre-processed in chunks'
        }

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Enhanced Video Processor with Chunk-based Architecture')
    parser.add_argument('--midi-json', required=True, help='Path to MIDI JSON file')
    parser.add_argument('--video-files-json', required=True, help='Path to video files JSON')
    parser.add_argument('--output-path', required=True, help='Output video path')
    parser.add_argument('--performance-mode', action='store_true', default=True, help='Enable performance mode')
    parser.add_argument('--memory-limit', type=int, default=4, help='Memory limit in GB')
    
    args = parser.parse_args()
    
    # Load input data
    try:
        with open(args.midi_json, 'r') as f:
            midi_data = json.load(f)
        with open(args.video_files_json, 'r') as f:
            video_files = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load input files: {e}")
        return False
    
    # Initialize processor
    processor = EnhancedVideoProcessor(
        performance_mode=args.performance_mode,
        memory_limit_gb=args.memory_limit
    )
    
    # Validate inputs
    if not processor.validate_input_files(args.midi_json, args.video_files_json):
        logger.error("Input validation failed")
        return False
    
    # Process videos using chunk-based approach
    logger.info("Starting chunk-based video processing...")
    success = processor.process_videos(midi_data, video_files, args.output_path)
    
    if success:
        logger.info(f"SUCCESS: Video processing completed successfully: {args.output_path}")
        
        # Show performance summary
        summary = processor.get_performance_summary()
        logger.info(f"Performance summary: {summary}")
    else:
        logger.error("ERROR: Video processing failed")
    
    # Cleanup
    processor.cleanup()
    
    return success

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
