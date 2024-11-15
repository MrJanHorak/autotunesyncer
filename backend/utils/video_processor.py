# import json
# import sys
# import subprocess
# import logging
# import os
# import tempfile
# import numpy as np  # Add this import
# from typing import List, Dict
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import math
# import ctypes
# from ctypes import wintypes
# import shutil  # Add this import for finding ffmpeg executable

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# def build_filter_complex(clips_data: List[Dict], duration: float) -> str:
#     """Generate FFmpeg filter complex string for all clips."""
#     if not clips_data:
#         return ""
    
#     # Process clips in smaller batches to keep filter string length manageable
#     batch_size = 5
#     filters = []
#     current = "[0:v]"
    
#     for batch_idx in range(0, len(clips_data), batch_size):
#         batch = clips_data[batch_idx:batch_idx + batch_size]
        
#         # Process each clip in the batch
#         for i, clip in enumerate(batch, 1):
#             idx = batch_idx + i
#             scale_out = f"[s{idx}]"
#             filters.append(f"[{idx}:v]scale={clip['width']}:{clip['height']}{scale_out}")
            
#             next_out = f"[v{idx}]"
#             if batch_idx + i == len(clips_data):  # Last clip overall
#                 next_out = "[v]"
            
#             filters.append(
#                 f"{current}{scale_out}overlay="
#                 f"x={clip['x']}:y={clip['y']}"
#                 f":enable='between(t,{clip['start_time']},{clip['start_time'] + clip['duration']})'"
#                 f"{next_out}"
#             )
#             current = next_out
    
#     return ";".join(filters)

# def get_short_path_name(long_name: str) -> str:
#     """Get Windows 8.3 short path."""
#     try:
#         buffer = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
#         GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
#         GetShortPathNameW(long_name, buffer, wintypes.MAX_PATH)
#         return buffer.value
#     except:
#         return long_name

# def create_temp_dir(base_path: str) -> str:
#     """Create a temporary directory with shortest possible path."""
#     try:
#         # Use drive root for shortest possible path
#         drive = os.path.splitdrive(base_path)[0]
#         temp_dir = os.path.join(drive + os.sep, 't')
#         if not os.path.exists(temp_dir):
#             os.makedirs(temp_dir)
#         return temp_dir
#     except Exception as e:
#         logging.error(f"Error creating temp directory: {e}")
#         raise

# def setup_working_directory(base_path: str) -> str:
#     """Create minimal working directory structure."""
#     drive = os.path.splitdrive(base_path)[0]
#     work_dir = drive + os.sep + 'x'  # Single character directory
#     if os.path.exists(work_dir):
#         for f in os.listdir(work_dir):
#             try:
#                 os.remove(os.path.join(work_dir, f))
#             except:
#                 pass
#     else:
#         os.makedirs(work_dir)
#     return work_dir

# def copy_to_work_dir(files: Dict[str, Dict], work_dir: str) -> Dict[str, str]:
#     """Copy source files to working directory with minimal names."""
#     file_map = {}
#     for idx, (track_id, track_data) in enumerate(files.items()):
#         if not track_data.get('notes'):
#             continue
#         src = track_data['path']
#         dst = os.path.join(work_dir, f"i{idx}.mp4")
#         import shutil
#         shutil.copy2(src, dst)
#         file_map[track_id] = dst
#     return file_map

# def execute_ffmpeg_command(ffmpeg_cmd: List[str]) -> None:
#     """Execute FFmpeg command and log outputs."""
#     try:
#         ffmpeg_path = shutil.which('ffmpeg')
#         if not ffmpeg_path:
#             raise Exception("FFmpeg executable not found in system PATH.")

#         ffmpeg_cmd[0] = ffmpeg_path

#         # Log the FFmpeg command
#         logging.debug(f"Executing FFmpeg command: {' '.join(ffmpeg_cmd)}")

#         process = subprocess.run(
#             ffmpeg_cmd,
#             capture_output=True,
#             text=True,
#             shell=False
#         )

#         # Log stdout and stderr
#         logging.debug(f"FFmpeg stdout: {process.stdout}")
#         logging.debug(f"FFmpeg stderr: {process.stderr}")

#         if process.returncode != 0:
#             raise Exception(f"FFmpeg failed: {process.stderr.strip()}")

#     except Exception as e:
#         logging.error(f"Error executing FFmpeg command: {e}")
#         raise

# def process_segment(segment_clips: List[Dict], duration: float, segment_start: float, 
#                    segment_duration: float, temp_dir: str, segment_idx: int) -> str:
#     """Process a single segment of video clips with pitch shifting."""
#     try:
#         # Group clips by track for better processing
#         tracks = {}
#         for clip in segment_clips:
#             track_id = clip['track_id']
#             if track_id not in tracks:
#                 tracks[track_id] = []
#             tracks[track_id].append(clip)

#         # Process each track separately first
#         track_outputs = []
#         for track_id, track_clips in tracks.items():
#             track_out = process_track(track_clips, segment_duration, segment_start, temp_dir, f"{segment_idx}_{track_id}")
#             if track_out:
#                 track_outputs.append(track_out)

#         # Combine track outputs
#         out_file = f"{temp_dir}{os.sep}{segment_idx:02d}.mp4"
#         combine_tracks(track_outputs, out_file, segment_duration)
        
#         # Cleanup track files
#         for file in track_outputs:
#             try:
#                 os.remove(file)
#             except:
#                 pass

#         return out_file

#     except Exception as e:
#         logging.error(f"Segment processing failed: {e}")
#         raise

# def process_track(track_clips: List[Dict], duration: float, segment_start: float, 
#                  temp_dir: str, track_id: str) -> str:
#     try:
#         CHUNK_SIZE = 5
#         chunk_outputs = []
        
#         for i in range(0, len(track_clips), CHUNK_SIZE):
#             chunk = track_clips[i:i + CHUNK_SIZE]
#             chunk_out = f"{temp_dir}{os.sep}chunk_{track_id}_{i}.mp4"
            
#             # Base command
#             base_cmd = [
#                 'ffmpeg', '-y',
#                 '-f', 'lavfi', '-i', f'color=black:s={chunk[0]["width"]}x{chunk[0]["height"]}:d={duration}:r=30',
#                 '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo:d={duration}'
#             ]
            
#             # Add clip inputs
#             for clip in chunk:
#                 base_cmd.extend(['-i', clip['source_path']])

#             # Build separate video and audio filter chains
#             video_chain = ["[0:v]null[v0]"]  # Start with black video
#             audio_chain = []
#             current = "v0"
            
#             # Process clips
#             for idx, clip in enumerate(chunk, 1):
#                 clip_time = clip['start_time'] - segment_start
#                 if clip_time < 0 or clip_time >= duration:
#                     continue
                
#                 # Video chain
#                 vout = f"v{idx}"
#                 video_chain.append(
#                     f"[{idx}:v]scale={clip['width']}:{clip['height']}[s{idx}];" +
#                     f"[{current}][s{idx}]overlay=x=0:y=0:enable='between(t,{clip_time},{clip_time + clip['duration']})'[{vout}]"
#                 )
#                 current = vout
                
#                 # Audio chain - keep separate from video
#                 if not clip.get('is_drum', False):
#                     speed_factor = pow(2, (clip['midi_note'] - 60) / 12)
#                     audio_chain.append(
#                         f"[{idx}:a]atrim=0:{clip['duration']},asetpts=PTS-STARTPTS," +
#                         f"asetrate=44100*{speed_factor},aresample=44100," +
#                         f"adelay={int(clip_time*1000)}|{int(clip_time*1000)}," +
#                         f"volume={clip.get('velocity', 1.0)}[a{idx}]"
#                     )

#             # Finalize chains
#             filters = []
            
#             # Add video chain
#             filters.extend(video_chain)
#             filters.append(f"[{current}]null[outv]")  # Ensure final video output
            
#             # Add audio chain and mixer
#             if audio_chain:
#                 filters.extend(audio_chain)
#                 audio_mix_inputs = ''.join(f'[a{i}]' for i in range(1, len(audio_chain) + 1))
#                 if audio_mix_inputs:
#                     filters.append(f"{audio_mix_inputs}amix=inputs={len(audio_chain)}:dropout_transition=0[outa]")
            
#             # Build filter complex
#             filter_complex = ';'.join(filters)
#             base_cmd.extend(['-filter_complex', filter_complex])
            
#             # Map outputs
#             base_cmd.extend(['-map', '[outv]'])
#             if audio_chain:
#                 base_cmd.extend(['-map', '[outa]'])
#             else:
#                 base_cmd.extend(['-map', '1:a'])
            
#             # Output settings
#             base_cmd.extend([
#                 '-c:v', 'h264_nvenc',
#                 '-c:a', 'aac',
#                 '-ar', '44100',
#                 '-b:a', '192k',
#                 chunk_out
#             ])
            
#             # Execute and validate
#             execute_ffmpeg_command(base_cmd)
#             if os.path.exists(chunk_out):
#                 chunk_outputs.append(chunk_out)

#         # Combine chunks if needed
#         if len(chunk_outputs) > 1:
#             final_output = f"{temp_dir}{os.sep}track_{track_id}.mp4"
#             combine_chunks(chunk_outputs, final_output)
#             return final_output
#         elif chunk_outputs:
#             return chunk_outputs[0]
        
#         return None

#     except Exception as e:
#         logging.error(f"Track processing failed: {e}")
#         raise

# def combine_chunks(chunk_files: List[str], output: str) -> None:
#     """Combine multiple chunk files."""
#     try:
#         # Create concat file
#         concat_file = os.path.splitext(output)[0] + '_concat.txt'
#         with open(concat_file, 'w') as f:
#             for file in chunk_files:
#                 f.write(f"file '{file}'\n")
        
#         # Combine chunks
#         cmd = [
#             'ffmpeg', '-y',
#             '-f', 'concat',
#             '-safe', '0',
#             '-i', concat_file,
#             '-c', 'copy',
#             output
#         ]
        
#         execute_ffmpeg_command(cmd)
        
#         # Cleanup concat file
#         try:
#             os.remove(concat_file)
#         except:
#             pass
            
#     except Exception as e:
#         logging.error(f"Error combining chunks: {e}")
#         raise

# def combine_tracks(track_files: List[str], output: str, duration: float) -> None:
#     """Combine multiple track videos into final output."""
#     try:
#         if not track_files:
#             return

#         cmd = ['ffmpeg', '-y']
        
#         # Add inputs
#         for file in track_files:
#             cmd.extend(['-i', file])
        
#         # Create overlay chain
#         filters = []
#         current = '[0:v]'
#         audio_inputs = []
        
#         for i in range(1, len(track_files)):
#             next_out = '[v]' if i == len(track_files) - 1 else f'[v{i}]'
#             filters.append(f"{current}[{i}:v]overlay=shortest=1{next_out}")
#             current = next_out
#             audio_inputs.append(f"[{i}:a]")
        
#         # Add audio mixing
#         if len(track_files) > 1:
#             filters.append(f"[0:a]{(''.join(audio_inputs))}amix=inputs={len(track_files)}:dropout_transition=0[a]")
        
#         # Add filter complex
#         if filters:
#             cmd.extend(['-filter_complex', ';'.join(filters)])
#             cmd.extend(['-map', '[v]'])
#             if len(track_files) > 1:
#                 cmd.extend(['-map', '[a]'])
#             else:
#                 cmd.extend(['-map', '0:a'])
        
#         # Add output options
#         cmd.extend([
#             '-c:v', 'h264_nvenc',
#             '-c:a', 'aac',
#             '-ar', '44100',
#             '-b:a', '192k',
#             '-t', str(duration),
#             output
#         ])
        
#         execute_ffmpeg_command(cmd)

#     except Exception as e:
#         logging.error(f"Track combination failed: {e}")
#         raise

# def process_large_segment(segment_clips: List[Dict], duration: float, segment_start: float,
#                           segment_duration: float, temp_dir: str, segment_idx: int, 
#                           batch_size: int) -> str:
#     """Process a large segment by breaking it into smaller batches."""
#     try:
#         # Process in batches
#         batch_files = []
#         for i in range(0, len(segment_clips), batch_size):
#             batch = segment_clips[i:i + batch_size]
#             batch_output = os.path.join(temp_dir, f"{segment_idx:02d}_{i // batch_size}.mp4")

#             # Process batch
#             ffmpeg_cmd = [
#                 'ffmpeg', '-y',
#                 '-hwaccel', 'cuda',
#                 '-hwaccel_output_format', 'cuda',
#                 '-f', 'lavfi',
#                 '-i', f'color=c=black:s=960x720:r=30:d={segment_duration}'
#             ]

#             # Add batch inputs
#             for clip in batch:
#                 ffmpeg_cmd.extend(['-i', clip['source_path']])

#             # Add filter complex for batch
#             filter_complex = build_filter_complex(batch, segment_duration)
#             filter_complex_file = None
#             with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as filter_file:
#                 filter_file.write(filter_complex)
#                 filter_complex_file = filter_file.name

#             ffmpeg_cmd.extend([
#                 '-filter_complex', f"@{filter_complex_file}",
#                 '-map', '[v]',
#                 '-c:v', 'h264_nvenc',
#                 '-preset', 'p7',
#                 batch_output
#             ])

#             process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             stdout, stderr = process.communicate()

#             # Clean up the filter complex temporary file
#             if filter_complex_file and os.path.exists(filter_complex_file):
#                 os.remove(filter_complex_file)

#             if process.returncode != 0:
#                 raise Exception(f"Batch processing failed: {stderr.decode()}")

#             batch_files.append(batch_output)

#         # Combine batch results
#         final_output = os.path.join(temp_dir, f"{segment_idx:02d}.mp4")
#         concat_file = os.path.join(temp_dir, 'concat.txt')
#         with open(concat_file, 'w') as f:
#             for file in batch_files:
#                 f.write(f"file '{file}'\n")

#         combine_cmd = [
#             'ffmpeg', '-y',
#             '-f', 'concat',
#             '-safe', '0',
#             '-i', concat_file,
#             '-c', 'copy',
#             final_output
#         ]

#         process = subprocess.Popen(combine_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         stdout, stderr = process.communicate()

#         # Cleanup batch files
#         for file in batch_files:
#             try:
#                 os.remove(file)
#             except:
#                 pass

#         # Clean up the concat file
#         if os.path.exists(concat_file):
#             os.remove(concat_file)

#         if process.returncode != 0:
#             raise Exception(f"Combining batches failed: {stderr.decode()}")

#         return final_output

#     except Exception as e:
#         logging.error(f"Error processing large segment: {e}")
#         raise

# def process_video_segments(midi_data: Dict, video_files: Dict, output_path: str) -> bool:
#     try:
#         # Create working directory and copy files
#         work_dir = setup_working_directory(output_path)
#         file_map = copy_to_work_dir(video_files, work_dir)
        
#         # Update paths in video_files
#         for track_id, track_data in video_files.items():
#             if track_id in file_map:
#                 track_data['path'] = file_map[track_id]

#         # Get duration from MIDI data
#         duration = float(midi_data.get('duration', 0))
#         if duration <= 0:
#             raise ValueError("Invalid duration in MIDI data")

#         logging.info(f"Total duration: {duration} seconds")

#         # Prepare clips data
#         clips_data = prepare_clips_data(video_files, duration)
#         if not clips_data:
#             raise ValueError("No valid clips to process")

#         # Process in 30-second segments
#         segment_duration = 30.0
#         num_segments = math.ceil(duration / segment_duration)
#         segment_files = []

#         logging.info(f"Processing {num_segments} segments")

#         for i in range(num_segments):
#             segment_start = i * segment_duration
#             current_duration = min(segment_duration, duration - segment_start)
            
#             logging.info(f"Processing segment {i+1}/{num_segments} "
#                         f"(start: {segment_start}, duration: {current_duration})")

#             try:
#                 segment_file = process_segment(
#                     clips_data,
#                     duration,
#                     segment_start,
#                     current_duration,
#                     work_dir,
#                     i
#                 )
#                 if os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
#                     segment_files.append(segment_file)
#                 else:
#                     raise Exception(f"Invalid segment file: {segment_file}")
#             except Exception as e:
#                 logging.error(f"Failed to process segment {i}: {e}")
#                 raise

#         if not segment_files:
#             raise ValueError("No segments were successfully created")

#         # Concatenate segments
#         concat_file = os.path.join(work_dir, 'concat.txt')
#         with open(concat_file, 'w') as f:
#             for file in segment_files:
#                 f.write(f"file '{file}'\n")

#         concat_cmd = [
#             'ffmpeg', '-y',
#             '-f', 'concat',
#             '-safe', '0',
#             '-i', concat_file,
#             '-c', 'copy',
#             '-t', str(duration),  # Explicitly set final duration
#             output_path
#         ]

#         execute_ffmpeg_command(concat_cmd)

#         # Cleanup
#         for file in segment_files + [concat_file]:
#             try:
#                 os.remove(file)
#             except:
#                 pass

#         return True

#     except Exception as e:
#         logging.error(f"Video processing failed: {e}")
#         raise
#     finally:
#         # Cleanup working directory
#         if 'work_dir' in locals() and os.path.exists(work_dir):
#             shutil.rmtree(work_dir, ignore_errors=True)

# def prepare_clips_data(video_files: Dict, duration: float) -> List[Dict]:
#     """Prepare clips data with proper MIDI note mapping."""
#     clips_data = []
    
#     for track_id, track_data in video_files.items():
#         if not track_data.get('notes'):
#             continue

#         # Calculate track position in grid
#         total_tracks = len([t for t in video_files.values() if t.get('notes')])
#         track_idx = list(video_files.keys()).index(track_id)
        
#         # Calculate dimensions based on track count
#         if total_tracks == 1:
#             width, height = 960, 720
#             x, y = 0, 0
#         else:
#             cols = math.ceil(math.sqrt(total_tracks))
#             rows = math.ceil(total_tracks / cols)
#             width = 960 // cols
#             height = 720 // rows
#             x = (track_idx % cols) * width
#             y = (track_idx // cols) * height

#         # Process each note in the track
#         for note in track_data['notes']:
#             clips_data.append({
#                 'x': x,
#                 'y': y,
#                 'width': width,
#                 'height': height,
#                 'start_time': float(note['time']),
#                 'duration': float(note['duration']),
#                 'is_drum': track_data.get('isDrum', False),
#                 'velocity': note.get('velocity', 1.0),
#                 'source_path': track_data['path'],
#                 'midi_note': note['midi'],
#                 'track_id': track_id
#             })

#     # Sort by start time for efficient processing
#     clips_data.sort(key=lambda x: x['start_time'])
#     return clips_data

# def main():
#     if len(sys.argv) != 4:
#         print("Usage: python video_processor.py midi_data.json video_files.json output_path")
#         sys.exit(1)

#     midi_json_path = sys.argv[1]
#     video_files_json_path = sys.argv[2]
#     output_path = sys.argv[3]

#     try:
#         with open(midi_json_path, 'r') as f:
#             midi_data = json.load(f)
#         with open(video_files_json_path, 'r') as f:
#             video_files = json.load(f)
            
#         process_video_segments(midi_data, video_files, output_path)
#     except Exception as e:
#         print(f"Error in video processing: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()