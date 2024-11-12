import json
import sys
import subprocess
import logging
import os
import tempfile
import numpy as np  # Add this import
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import ctypes
from ctypes import wintypes
import shutil  # Add this import for finding ffmpeg executable

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def build_filter_complex(clips_data: List[Dict], duration: float) -> str:
    """Generate FFmpeg filter complex string for all clips."""
    if not clips_data:
        return ""
    
    # Process clips in smaller batches to keep filter string length manageable
    batch_size = 5
    filters = []
    current = "[0:v]"
    
    for batch_idx in range(0, len(clips_data), batch_size):
        batch = clips_data[batch_idx:batch_idx + batch_size]
        
        # Process each clip in the batch
        for i, clip in enumerate(batch, 1):
            idx = batch_idx + i
            scale_out = f"[s{idx}]"
            filters.append(f"[{idx}:v]scale={clip['width']}:{clip['height']}{scale_out}")
            
            next_out = f"[v{idx}]"
            if batch_idx + i == len(clips_data):  # Last clip overall
                next_out = "[v]"
            
            filters.append(
                f"{current}{scale_out}overlay="
                f"x={clip['x']}:y={clip['y']}"
                f":enable='between(t,{clip['start_time']},{clip['start_time'] + clip['duration']})'"
                f"{next_out}"
            )
            current = next_out
    
    return ";".join(filters)

def get_short_path_name(long_name: str) -> str:
    """Get Windows 8.3 short path."""
    try:
        buffer = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW(long_name, buffer, wintypes.MAX_PATH)
        return buffer.value
    except:
        return long_name

def create_temp_dir(base_path: str) -> str:
    """Create a temporary directory with shortest possible path."""
    try:
        # Use drive root for shortest possible path
        drive = os.path.splitdrive(base_path)[0]
        temp_dir = os.path.join(drive + os.sep, 't')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        return temp_dir
    except Exception as e:
        logging.error(f"Error creating temp directory: {e}")
        raise

def setup_working_directory(base_path: str) -> str:
    """Create minimal working directory structure."""
    drive = os.path.splitdrive(base_path)[0]
    work_dir = drive + os.sep + 'x'  # Single character directory
    if os.path.exists(work_dir):
        for f in os.listdir(work_dir):
            try:
                os.remove(os.path.join(work_dir, f))
            except:
                pass
    else:
        os.makedirs(work_dir)
    return work_dir

def copy_to_work_dir(files: Dict[str, Dict], work_dir: str) -> Dict[str, str]:
    """Copy source files to working directory with minimal names."""
    file_map = {}
    for idx, (track_id, track_data) in enumerate(files.items()):
        if not track_data.get('notes'):
            continue
        src = track_data['path']
        dst = os.path.join(work_dir, f"i{idx}.mp4")
        import shutil
        shutil.copy2(src, dst)
        file_map[track_id] = dst
    return file_map

def execute_ffmpeg_command(ffmpeg_cmd: List[str]) -> None:
    """Execute FFmpeg command and log outputs."""
    try:
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            raise Exception("FFmpeg executable not found in system PATH.")

        ffmpeg_cmd[0] = ffmpeg_path

        # Log the FFmpeg command
        logging.debug(f"Executing FFmpeg command: {' '.join(ffmpeg_cmd)}")

        process = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            shell=False
        )

        # Log stdout and stderr
        logging.debug(f"FFmpeg stdout: {process.stdout}")
        logging.debug(f"FFmpeg stderr: {process.stderr}")

        if process.returncode != 0:
            raise Exception(f"FFmpeg failed: {process.stderr.strip()}")

    except Exception as e:
        logging.error(f"Error executing FFmpeg command: {e}")
        raise

def process_segment(
    segment_clips: List[Dict],
    duration: float,
    segment_start: float,
    segment_duration: float,
    temp_dir: str,
    segment_idx: int
) -> str:
    """Process a single segment of video clips."""
    try:
        out_file = f"{temp_dir}{os.sep}{segment_idx:02d}.mp4"
        logging.info(f"Processing segment {segment_idx}")

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f'color=black:s=960x720:d={segment_duration}'
        ]

        # Filter clips for this segment
        segment_clips = [
            clip for clip in segment_clips
            if (clip['start_time'] + clip['duration'] > segment_start and 
                clip['start_time'] < segment_start + segment_duration)
        ]

        if segment_clips:
            # Add inputs with minimal paths
            for clip in segment_clips:
                if not os.path.exists(clip['source_path']):
                    raise ValueError(f"Missing source: {clip['source_path']}")
                ffmpeg_cmd.extend(['-i', clip['source_path']])

            # Adjust timestamps
            for clip in segment_clips:
                clip['start_time'] = max(0, clip['start_time'] - segment_start)

            # Generate minimal filter
            filters = []
            last = "[0:v]"
            for i, clip in enumerate(segment_clips, 1):
                v = f"[v{i}]"
                filters.extend([
                    f"[{i}:v]scale={clip['width']}:{clip['height']}[s{i}]",
                    f"{last}[s{i}]overlay={clip['x']}:{clip['y']}:enable='between(t\\,{clip['start_time']}\\,{clip['start_time'] + clip['duration']})'{v if i < len(segment_clips) else ''}"
                ])
                last = v

            filter_complex_str = ';'.join(filters)

            # Write filter complex to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as filter_file:
                filter_file.write(filter_complex_str)
                filter_file_path = filter_file.name

            ffmpeg_cmd.extend([
                '-filter_complex_script', filter_file_path,
                '-c:v', 'h264_nvenc',
                '-preset', 'p7',
                '-rc', 'vbr',
                '-b:v', '5M',
                out_file
            ])

            # Clean up the filter complex temporary file after execution
            cleanup_filter_file = True
        else:
            # Correct FFmpeg command for empty segments
            ffmpeg_cmd.extend([
                '-c:v', 'h264_nvenc',
                '-t', str(segment_duration),
                out_file
            ])
            cleanup_filter_file = False

        # Execute the FFmpeg command
        execute_ffmpeg_command(ffmpeg_cmd)

        if not os.path.exists(out_file) or os.path.getsize(out_file) == 0:
            raise Exception("Output not created")

        # Clean up the filter complex temporary file if it was created
        if cleanup_filter_file and 'filter_file_path' in locals():
            if os.path.exists(filter_file_path):
                os.remove(filter_file_path)

        return out_file

    except Exception as e:
        logging.error(f"Segment {segment_idx} failed: {str(e)}")
        raise

def process_large_segment(segment_clips: List[Dict], duration: float, segment_start: float,
                          segment_duration: float, temp_dir: str, segment_idx: int, 
                          batch_size: int) -> str:
    """Process a large segment by breaking it into smaller batches."""
    try:
        # Process in batches
        batch_files = []
        for i in range(0, len(segment_clips), batch_size):
            batch = segment_clips[i:i + batch_size]
            batch_output = os.path.join(temp_dir, f"{segment_idx:02d}_{i // batch_size}.mp4")

            # Process batch
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-hwaccel', 'cuda',
                '-hwaccel_output_format', 'cuda',
                '-f', 'lavfi',
                '-i', f'color=c=black:s=960x720:r=30:d={segment_duration}'
            ]

            # Add batch inputs
            for clip in batch:
                ffmpeg_cmd.extend(['-i', clip['source_path']])

            # Add filter complex for batch
            filter_complex = build_filter_complex(batch, segment_duration)
            filter_complex_file = None
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as filter_file:
                filter_file.write(filter_complex)
                filter_complex_file = filter_file.name

            ffmpeg_cmd.extend([
                '-filter_complex_script', filter_complex_file,
                '-map', '[v]',
                '-c:v', 'h264_nvenc',
                '-preset', 'p7',
                batch_output
            ])

            process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            # Clean up the filter complex temporary file
            if filter_complex_file and os.path.exists(filter_complex_file):
                os.remove(filter_complex_file)

            if process.returncode != 0:
                raise Exception(f"Batch processing failed: {stderr.decode()}")

            batch_files.append(batch_output)

        # Combine batch results
        final_output = os.path.join(temp_dir, f"{segment_idx:02d}.mp4")
        concat_file = os.path.join(temp_dir, 'concat.txt')
        with open(concat_file, 'w') as f:
            for file in batch_files:
                f.write(f"file '{file}'\n")

        combine_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            final_output
        ]

        process = subprocess.Popen(combine_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Cleanup batch files
        for file in batch_files:
            try:
                os.remove(file)
            except:
                pass

        # Clean up the concat file
        if os.path.exists(concat_file):
            os.remove(concat_file)

        if process.returncode != 0:
            raise Exception(f"Combining batches failed: {stderr.decode()}")

        return final_output

    except Exception as e:
        logging.error(f"Error processing large segment: {e}")
        raise

def process_video_segments(midi_data: Dict, video_files: Dict, output_path: str) -> bool:
    work_dir = None
    try:
        # Create working directory and copy files
        work_dir = setup_working_directory(output_path)
        file_map = copy_to_work_dir(video_files, work_dir)
        
        # Update video_files with new minimal paths
        for track_id, track_data in video_files.items():
            if track_id in file_map:
                track_data['path'] = file_map[track_id]

        # Rest of processing remains the same, but use work_dir instead of creating new temp dir
        duration = float(midi_data.get('duration', 0))
        if duration <= 0:
            raise ValueError("Invalid duration in MIDI data")

        # Prepare clips data
        clips_data = prepare_clips_data(video_files, duration)
        
        if not clips_data:
            raise ValueError("No valid clips to process")

        # Split into 30-second segments (or adjust based on your needs)
        segment_duration = 30.0
        num_segments = math.ceil(duration / segment_duration)
        segment_files = []

        BATCH_THRESHOLD = 50  # Set a threshold for batch processing

        # Process segments sequentially instead of parallel for better stability
        for i in range(num_segments):
            try:
                segment_start = i * segment_duration
                current_duration = min(segment_duration, duration - segment_start)

                # Filter clips for this segment
                segment_clips = [
                    clip for clip in clips_data
                    if (clip['start_time'] + clip['duration'] > segment_start and 
                        clip['start_time'] < segment_start + current_duration)
                ]

                # Log the number of clips in the segment
                logging.debug(f"Segment {i}: {len(segment_clips)} clips")

                # Use process_large_segment if number of clips exceeds threshold
                if len(segment_clips) > BATCH_THRESHOLD:
                    segment_file = process_large_segment(
                        segment_clips,
                        duration,
                        segment_start,
                        current_duration,
                        work_dir,
                        i,
                        batch_size=5  # Adjust batch size as needed
                    )
                else:
                    segment_file = process_segment(
                        segment_clips,
                        duration,
                        segment_start,
                        current_duration,
                        work_dir,
                        i
                    )
                
                if os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
                    segment_files.append(segment_file)
                else:
                    logging.error(f"Invalid segment file created: {segment_file}")
                    
            except Exception as e:
                logging.error(f"Failed to process segment {i}: {e}")
                raise

        if not segment_files:
            raise ValueError("No segments were successfully created")

        # Concatenate segments
        if segment_files:
            # Ensure segments exist and are valid
            valid_segments = []
            for file in sorted(segment_files):
                if os.path.exists(file) and os.path.getsize(file) > 0:
                    valid_segments.append(file)
                else:
                    logging.warning(f"Skipping invalid segment file: {file}")

            if not valid_segments:
                raise ValueError("No valid segments to concatenate")

            # Create concat file with absolute paths
            concat_file = os.path.join(work_dir, 'list.txt')
            logging.info(f"Creating concat file at: {concat_file}")
            logging.info(f"Segments to concatenate: {valid_segments}")

            with open(concat_file, 'w') as f:
                for file in valid_segments:
                    abs_path = os.path.abspath(file)
                    f.write(f"file '{abs_path.replace(os.sep, '/')}'\n")

            # Execute concatenation with detailed logging
            concat_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                output_path
            ]

            logging.info(f"Running concat command: {' '.join(concat_cmd)}")
            process = subprocess.Popen(
                concat_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True  # Get string output instead of bytes
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                logging.error(f"Concatenation failed with error: {stderr}")
                raise Exception(f"Final concatenation failed: {stderr}")
            else:
                logging.info("Concatenation completed successfully")

            # Cleanup temp files and directory
            for file in segment_files + [concat_file]:
                try:
                    if os.path.exists(file):
                        os.remove(file)
                except:
                    pass
            try:
                os.rmdir(work_dir)
            except:
                pass

        return True

    finally:
        # Cleanup working directory
        if work_dir and os.path.exists(work_dir):
            for file in os.listdir(work_dir):
                try:
                    os.remove(os.path.join(work_dir, file))
                except:
                    pass
            try:
                os.rmdir(work_dir)
            except:
                pass

def prepare_clips_data(video_files: Dict, duration: float) -> List[Dict]:
    """Prepare clips data with optimized layout calculations."""
    clips_data = []
    
    # Process each track's notes
    for track_id, track_data in video_files.items():
        if not track_data.get('notes'):
            continue

        source_path = track_data['path']
        is_drum = track_data.get('isDrum', False)
        
        # Calculate clip dimensions based on track count
        total_tracks = len([t for t in video_files.values() if t.get('notes')])
        if total_tracks == 1:
            width, height = 960, 720
            x, y = 0, 0
        elif total_tracks == 2:
            width, height = 480, 720
            x = len(clips_data) * 480
            y = 0
        else:
            grid_size = max(2, int(np.ceil(np.sqrt(total_tracks))))
            width = 960 // grid_size
            height = 720 // grid_size
            x = (len(clips_data) % grid_size) * width
            y = (len(clips_data) // grid_size) * height

        # Add each note as a clip
        for note in track_data['notes']:
            clips_data.append({
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'start_time': float(note['time']),
                'duration': float(note['duration']),
                'is_drum': is_drum,
                'velocity': note.get('velocity', 1.0),
                'source_path': source_path
            })

    return clips_data

def main():
    if len(sys.argv) != 4:
        print("Usage: python video_processor.py midi_data.json video_files.json output_path")
        sys.exit(1)

    midi_json_path = sys.argv[1]
    video_files_json_path = sys.argv[2]
    output_path = sys.argv[3]

    try:
        with open(midi_json_path, 'r') as f:
            midi_data = json.load(f)
        with open(video_files_json_path, 'r') as f:
            video_files = json.load(f)
            
        process_video_segments(midi_data, video_files, output_path)
    except Exception as e:
        print(f"Error in video processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()