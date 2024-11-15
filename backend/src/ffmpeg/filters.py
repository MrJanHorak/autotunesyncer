from typing import List, Dict

def build_filter_complex(clips_data: List[Dict], duration: float) -> str:
    """Generate FFmpeg filter complex string for all clips."""
    if not clips_data:
        return ""
    
    batch_size = 5
    filters = []
    current = "[0:v]"
    
    for batch_idx in range(0, len(clips_data), batch_size):
        batch = clips_data[batch_idx:batch_idx + batch_size]
        
        for i, clip in enumerate(batch, 1):
            idx = batch_idx + i
            scale_out = f"[s{idx}]"
            filters.append(f"[{idx}:v]scale={clip['width']}:{clip['height']}{scale_out}")
            
            next_out = f"[v{idx}]"
            if batch_idx + i == len(clips_data):
                next_out = "[v]"
            
            filters.append(
                f"{current}{scale_out}overlay="
                f"x={clip['x']}:y={clip['y']}"
                f":enable='between(t,{clip['start_time']},{clip['start_time'] + clip['duration']})'"
                f"{next_out}"
            )
            current = next_out
    
    return ";".join(filters)