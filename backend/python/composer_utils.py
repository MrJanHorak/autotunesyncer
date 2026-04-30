"""
composer_utils.py — Utility classes and helpers for VideoComposer.

Extracted from video_composer.py to aid navigation and maintainability.
VideoComposer and all callers import these from this module.
"""

import logging
import subprocess
import threading
import weakref
from contextlib import contextmanager

import torch


class GPUStreamManager:
    """Manages multiple CUDA streams for parallel GPU operations."""

    def __init__(self, num_streams=4):
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream = 0

    def get_stream(self):
        """Return next stream in round-robin order."""
        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % len(self.streams)
        return stream

    def synchronize_all(self):
        """Wait for all streams to complete."""
        for stream in self.streams:
            torch.cuda.synchronize(stream.device)


def gpu_subprocess_run(cmd, **kwargs):
    """
    Run a subprocess, adding GPU encoding for simple FFmpeg commands when CUDA
    is available. Falls back to plain subprocess.run on any failure.
    """
    try:
        # Complex note-triggered filter graphs — always CPU
        if any('trim=' in str(arg) and 'overlay=' in str(arg) for arg in cmd):
            return subprocess.run(cmd, **kwargs)

        if cmd[0] == 'ffmpeg' and torch.cuda.is_available():
            # Concat demuxer commands should not be modified
            if '-f' in cmd and 'concat' in cmd:
                return subprocess.run(cmd, **kwargs)

            # Only try GPU encoding when codec isn't already specified
            if '-c:v' not in cmd:
                gpu_cmd = cmd.copy()
                for i, arg in enumerate(cmd):
                    if arg.endswith('.mp4') and not arg.startswith('-'):
                        gpu_cmd[i:i] = ['-c:v', 'h264_nvenc', '-preset', 'fast']
                        break
                try:
                    return subprocess.run(gpu_cmd, **kwargs)
                except subprocess.CalledProcessError as exc:
                    logging.warning(f"GPU encode failed ({exc}), retrying with CPU")

        return subprocess.run(cmd, **kwargs)
    except Exception as exc:
        logging.error(f"gpu_subprocess_run error: {exc}")
        return subprocess.run(cmd, **kwargs)


class ClipPool:
    """Bounded semaphore pool to limit concurrent open video clips."""

    def __init__(self, max_size=8):
        self.semaphore = threading.BoundedSemaphore(max_size)
        self.clips = weakref.WeakSet()

    @contextmanager
    def acquire(self):
        self.semaphore.acquire()
        try:
            clip = None
            yield clip
        finally:
            if clip:
                clip.close()
            self.semaphore.release()


class ClipManager:
    """Context-manager wrapper that closes video clips on exit."""

    def __init__(self):
        self.active_clips = weakref.WeakSet()

    @contextmanager
    def managed_clip(self, clip):
        try:
            self.active_clips.add(clip)
            yield clip
        finally:
            try:
                clip.close()
            except Exception:
                pass
            self.active_clips.discard(clip)


class VideoComposerConfig:
    """Default configuration constants for VideoComposer."""

    def __init__(self):
        self.CHUNK_DURATION = 16
        self.OVERLAP_DURATION = 1
        self.CROSSFADE_DURATION = 0.5
        self.MIN_VIDEO_DURATION = 1.0
        self.DURATION = 1.0
        self.VOLUME_MULTIPLIERS = {
            'drums': 1.0,
            'instruments': 1.0,
        }


def get_system_metrics():
    """Return basic system resource metrics (CPU/RAM/disk)."""
    try:
        import psutil
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': (
                psutil.disk_usage('C:').percent
                if hasattr(psutil, 'disk_usage') and __import__('os').name == 'nt'
                else psutil.disk_usage('/').percent
            ),
        }
    except ImportError:
        return {'cpu_percent': 0, 'memory_percent': 0, 'disk_percent': 0}
