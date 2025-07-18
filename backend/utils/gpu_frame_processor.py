import torch
import torchvision
import torchvision.transforms as T
import os
from torchvision.io import read_video, write_video

def process_video_on_gpu(input_path, output_path, frame_fn=None, batch_size=32, device=None):
    """
    Loads a video, processes frames on GPU, and writes output video.
    frame_fn: function(tensor[B, C, H, W]) -> tensor[B, C, H, W]
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Processing {input_path} on {device}")
    video, audio, info = read_video(input_path)
    video = video.float().to(device) / 255.0  # [T, H, W, C], 0-1
    video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
    processed = []
    for i in range(0, video.shape[0], batch_size):
        batch = video[i:i+batch_size]
        if frame_fn:
            batch = frame_fn(batch)
        processed.append(batch)
    processed = torch.cat(processed, dim=0)
    processed = (processed * 255).byte().permute(0, 2, 3, 1).cpu()  # [T, H, W, C]
    write_video(output_path, processed, info['video_fps'])
    print(f"Saved processed video to {output_path}")

# Example: invert colors
if __name__ == '__main__':
    def invert(batch):
        return 1.0 - batch
    process_video_on_gpu('input.mp4', 'output_inverted.mp4', frame_fn=invert)
