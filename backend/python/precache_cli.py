#!/usr/bin/env python3
"""
CLI entrypoint for warming the autotune cache for a single video file.

Usage:
    echo '{"video_path": "...", "midi_notes": [60, 67, 72]}' | python precache_cli.py

Reads JSON from stdin so the Node.js parent can pass arbitrary-length note lists
without worrying about command-line argument length limits.

Exit code 0 on success, 1 on error.
"""

import os
import sys
import json
import logging

# Ensure imports resolve from the same directory
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

def main():
    try:
        data = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON input: {e}")
        sys.exit(1)

    video_path = data.get('video_path')
    midi_notes = data.get('midi_notes', [])

    if not video_path or not os.path.exists(video_path):
        logging.error(f"video_path not found: {video_path}")
        sys.exit(1)

    if not midi_notes:
        logging.warning("No MIDI notes provided, nothing to pre-cache.")
        sys.exit(0)

    # Deduplicate notes
    unique_notes = sorted(set(int(n) for n in midi_notes))
    logging.info(f"Pre-caching {len(unique_notes)} notes for {os.path.basename(video_path)}: {unique_notes}")

    try:
        from optimized_autotune_cache import OptimizedAutotuneCache
        cache = OptimizedAutotuneCache(max_workers=2)
        for note in unique_notes:
            result = cache.get_tuned_video(video_path, note)
            if result:
                logging.info(f"  ✅ Note {note} → {os.path.basename(result)}")
            else:
                logging.warning(f"  ⚠️  Note {note} → precache failed")
    except Exception as e:
        logging.error(f"Pre-cache failed: {e}")
        sys.exit(1)

    print(f"PRECACHE_DONE:{len(unique_notes)}", flush=True)
    sys.exit(0)

if __name__ == '__main__':
    main()
