import numpy as np
import cv2
from PyNvCodec import PyNvDecoder, PyNvEncoder
import torch

class StereoConverter:
    def __init__(self, input_video, depth_video, output_path):
        self.video_decoder = PyNvDecoder(input_video)
        self.depth_decoder = PyNvDecoder(depth_video)
        
        # Get video properties
        self.width = self.video_decoder.Width()
        self.height = self.video_decoder.Height()
        
        # Initialize encoder for side-by-side 3D
        self.encoder = PyNvEncoder(
            (self.width * 2, self.height),  # Double width for side-by-side
            fps=30
        )
        
        # Move processing to GPU
        self.device = torch.device('cuda')
        
    def generate_stereo_frame(self, frame, depth):
        # Convert to GPU tensors
        frame_tensor = torch.from_numpy(frame).to(self.device)
        depth_tensor = torch.from_numpy(depth).to(self.device)
        
        # Generate right view using depth information
        right_view = self.warp_frame(frame_tensor, depth_tensor)
        
        # Combine views side by side
        stereo_frame = torch.cat([frame_tensor, right_view], dim=1)
        return stereo_frame.cpu().numpy()
    
    def warp_frame(self, frame, depth):
        # Implement depth-based warping on GPU
        displacement = depth * 0.1  # Adjust stereo strength
        grid = self.generate_displacement_grid(displacement)
        return torch.nn.functional.grid_sample(frame, grid)
    
    def process(self):
        while True:
            frame = self.video_decoder.read()
            depth = self.depth_decoder.read()
            
            if frame is None or depth is None:
                break
                
            stereo_frame = self.generate_stereo_frame(frame, depth)
            self.encoder.encode(stereo_frame)
