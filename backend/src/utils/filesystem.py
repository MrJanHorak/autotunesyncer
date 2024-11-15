import os
import shutil
import ctypes
from ctypes import wintypes
import logging
from typing import Dict

def get_short_path_name(long_name: str) -> str:
    """Get Windows 8.3 short path."""
    try:
        buffer = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW(long_name, buffer, wintypes.MAX_PATH)
        return buffer.value
    except:
        return long_name

def setup_working_directory(base_path: str) -> str:
    """Create minimal working directory structure."""
    drive = os.path.splitdrive(base_path)[0]
    work_dir = drive + os.sep + 'x'
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
        shutil.copy2(src, dst)
        file_map[track_id] = dst
    return file_map