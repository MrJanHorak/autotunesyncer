"""
Shared FFmpeg encoding profile matrix for AutoTuneSyncer.

Returns FFmpeg argument lists for consistent quality across all encode sites.
Bump ENCODE_PROFILE_VERSION (and the autotune cache CACHE_VERSION) whenever
the chosen settings change so cached files get regenerated automatically.

Profile matrix
--------------
                  GPU (h264_nvenc)          CPU (libx264)
  preview         preset p1, cq 28          preset ultrafast, crf 28
  production      preset p3, cq 23          preset fast, crf 22
  master          preset p3, cq 18          preset medium, crf 18
  (precache mux)  ↑ master quality ↑
"""

import os
import logging

ENCODE_PROFILE_VERSION = "v1"


def _has_nvenc() -> bool:
    """Lightweight check: return True when h264_nvenc is available."""
    try:
        import subprocess
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5
        )
        return "h264_nvenc" in r.stdout
    except Exception:
        return False


# Module-level lazy cache so we only probe FFmpeg once per process
_nvenc_available: bool | None = None


def nvenc_available() -> bool:
    global _nvenc_available
    if _nvenc_available is None:
        _nvenc_available = _has_nvenc()
    return _nvenc_available


def get_video_encode_args(
    mode: str = "production",
    use_gpu: bool | None = None,
) -> list[str]:
    """
    Return a list of FFmpeg arguments for the video codec only (no -i, no output).

    Parameters
    ----------
    mode : "preview" | "production" | "master"
        Quality tier.  "master" is used for pre-cache intermediate files that
        will be re-encoded once during composition.
    use_gpu : bool | None
        Force GPU or CPU.  None = auto-detect.

    Returns
    -------
    list[str]
        e.g. ["-c:v", "h264_nvenc", "-preset", "p3", "-rc", "vbr", "-cq", "23",
               "-pix_fmt", "yuv420p"]
    """
    if use_gpu is None:
        use_gpu = nvenc_available()

    if mode == "preview":
        if use_gpu:
            return [
                "-c:v", "h264_nvenc",
                "-preset", "p1",
                "-rc", "vbr",
                "-cq", "28",
                "-pix_fmt", "yuv420p",
            ]
        else:
            return [
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "28",
                "-pix_fmt", "yuv420p",
            ]
    elif mode == "production":
        if use_gpu:
            return [
                "-c:v", "h264_nvenc",
                "-preset", "p3",
                "-rc", "vbr",
                "-cq", "23",
                "-pix_fmt", "yuv420p",
            ]
        else:
            return [
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22",
                "-pix_fmt", "yuv420p",
            ]
    elif mode == "master":
        # High quality for intermediate/precache files that go through one more encode
        if use_gpu:
            return [
                "-c:v", "h264_nvenc",
                "-preset", "p3",
                "-rc", "vbr",
                "-cq", "18",
                "-pix_fmt", "yuv420p",
            ]
        else:
            return [
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
            ]
    else:
        logging.warning(f"[ffmpeg_profiles] Unknown mode '{mode}', defaulting to production")
        return get_video_encode_args("production", use_gpu=use_gpu)
