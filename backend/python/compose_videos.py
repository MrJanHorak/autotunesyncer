# compose_videos.py
import json
import logging
from pathlib import Path
from audio_processor import process_track_videos, AudioVideoProcessor
from video_composer import compose_from_processor_output

if __name__ == "__main__":
    import argparse
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('config_path', help='Path to config JSON file')
        args = parser.parse_args()
        
        with open(args.config_path) as f:
            config = json.load(f)
            
        processor = AudioVideoProcessor()
        processor.setup_temp_directory()
        
        result = process_track_videos(
            config.get('tracks', {}), 
            config.get('videos', {}),
            processor
        )
        
        output_path = Path(processor.videos_dir).parent / f"final_composition_{processor.session_id}.mp4"
        
        composition_result = compose_from_processor_output(
            {
                'processed_videos_dir': processor.videos_dir,
                'tracks': config['tracks'],
                'processed_files': result
            },
            str(output_path)
        )
        
        print(json.dumps({
            'success': True,
            'data': {
                'processed': result,
                'composition': composition_result
            }
        }), flush=True)
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print(json.dumps({
            'success': False,
            'error': str(e)
        }), flush=True)