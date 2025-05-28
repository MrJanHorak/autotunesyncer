#!/usr/bin/env python3
"""
Enhanced Processor Adapter
Bridges the legacy video processing API with the new enhanced video processor
"""

import sys
import json
import logging
import os
from pathlib import Path

# Add the utils directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from video_processor import EnhancedVideoProcessor
    from preprocess_videos import VideoPreprocessor
    ENHANCED_PROCESSOR_AVAILABLE = True
except ImportError:
    logging.warning("Enhanced video processor not available, falling back to legacy")
    ENHANCED_PROCESSOR_AVAILABLE = False

def preprocess_videos_enhanced(video_files, options=None):
    """Preprocess videos using the enhanced preprocessor"""
    if not options:
        options = {}
    
    try:
        preprocessor = VideoPreprocessor(
            performance_mode=options.get('performance_mode', True),
            max_workers=options.get('max_workers', 4)
        )
        
        # Convert video_files dict to list format
        video_list = []
        for key, path in video_files.items():
            if isinstance(path, str) and os.path.exists(path):
                video_list.append({
                    'input': path,
                    'output': f'preprocessed_{key}.mp4'
                })
        
        if not video_list:
            return video_files  # No preprocessing needed
        
        # Create output directory
        output_dir = options.get('output_dir', os.path.join(os.path.dirname(__file__), 'preprocessed'))
        os.makedirs(output_dir, exist_ok=True)
        
        # Process videos
        results = preprocessor.preprocess_videos_batch(
            video_list, 
            output_dir, 
            options.get('target_size')
        )
        
        # Update video_files with preprocessed paths
        processed_files = {}
        for result in results:
            if result['success']:
                # Map back to original key
                original_key = None
                for key, path in video_files.items():
                    if path == result['input']:
                        original_key = key
                        break
                
                if original_key:
                    processed_files[original_key] = result['output']
                else:
                    # Fallback to original if mapping fails
                    processed_files[list(video_files.keys())[len(processed_files)]] = result['output']
            else:
                logging.warning(f"Preprocessing failed for {result['input']}, using original")
                # Find original key and keep original path
                for key, path in video_files.items():
                    if path == result['input']:
                        processed_files[key] = path
                        break
        
        return processed_files
        
    except Exception as e:
        logging.error(f"Enhanced preprocessing failed: {e}")
        return video_files  # Return original files on failure

def process_with_enhanced_processor(midi_data, video_files, output_path):
    """Process using the enhanced video processor"""
    try:
        processor = EnhancedVideoProcessor(
            performance_mode=True,
            memory_limit_gb=4,
            parallel_tracks=4
        )
        
        success = processor.process_videos(midi_data, video_files, output_path)
        
        if success:
            return {
                'success': True,
                'output_path': output_path,
                'processor': 'enhanced'
            }
        else:
            raise Exception("Enhanced processor failed")
            
    except Exception as e:
        logging.error(f"Enhanced processor failed: {e}")
        raise

def process_with_legacy_processor(midi_data, video_files, output_path):
    """Fallback to legacy processing"""
    try:
        # Import legacy components
        from audio_processor import process_track_videos, AudioVideoProcessor
        from video_composer import compose_from_processor_output
        
        processor = AudioVideoProcessor()
        processor.setup_temp_directory()
        
        # Process tracks
        result = process_track_videos(
            {'tracks': midi_data.get('tracks', [])},
            video_files,
            processor
        )
        
        # Compose final video
        composition_result = compose_from_processor_output(
            {
                'processed_videos_dir': processor.videos_dir,
                'tracks': midi_data,
                'processed_files': result
            },
            output_path
        )
        
        if composition_result:
            return {
                'success': True,
                'output_path': output_path,
                'processor': 'legacy'
            }
        else:
            raise Exception("Legacy processor failed")
            
    except Exception as e:
        logging.error(f"Legacy processor failed: {e}")
        raise

def main():
    """Main processing function with automatic fallback"""
    if len(sys.argv) < 4:
        print("Usage: python enhanced_processor_adapter.py midi_data.json video_files.json output_path [--legacy]")
        sys.exit(1)
    
    midi_json_path = sys.argv[1]
    video_files_json_path = sys.argv[2]
    output_path = sys.argv[3]
    force_legacy = '--legacy' in sys.argv
    
    try:
        # Load input data
        with open(midi_json_path, 'r') as f:
            midi_data = json.load(f)
        with open(video_files_json_path, 'r') as f:
            video_files = json.load(f)
        
        # Choose processor
        if force_legacy or not ENHANCED_PROCESSOR_AVAILABLE:
            logging.info("Using legacy processor")
            result = process_with_legacy_processor(midi_data, video_files, output_path)
        else:
            try:
                logging.info("Attempting enhanced processor")
                result = process_with_enhanced_processor(midi_data, video_files, output_path)
            except Exception as e:
                logging.warning(f"Enhanced processor failed, falling back to legacy: {e}")
                result = process_with_legacy_processor(midi_data, video_files, output_path)
        
        # Output result
        print(json.dumps(result), flush=True)
        
    except json.JSONDecodeError as e:
        error_result = {
            'success': False,
            'error': f"Invalid JSON format: {e}",
            'processor': 'none'
        }
        print(json.dumps(error_result), flush=True)
        sys.exit(1)
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'processor': 'failed'
        }
        print(json.dumps(error_result), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
