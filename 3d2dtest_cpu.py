import argparse
import cv2
import numpy as np
from tqdm import tqdm

def convert_to_stereo_3d(video_path, depth_path, output_path):
    # Open the input video and depth map video
    video = cv2.VideoCapture(video_path)
    depth_video = cv2.VideoCapture(depth_path)

    # Get the video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Create the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process each frame with progress bar
    for _ in tqdm(range(total_frames), desc='Processing Frames'):
        ret, frame = video.read()
        depth_ret, depth_frame = depth_video.read()

        if not ret or not depth_ret:
            break

        # Move both frame and depth map to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_depth = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        gpu_depth.upload(depth_frame)

        # Normalize depth map on the GPU
        gpu_depth_normalized = cv2.cuda.normalize(gpu_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Create left frame directly on GPU
        left_frame = gpu_frame

        # Initialize the right frame with a border
        right_frame = cv2.cuda.copyMakeBorder(gpu_frame, 0, 0, 0, int(width / 20), cv2.BORDER_REPLICATE)

        # Shift pixels in the right frame based on the depth map
        for i in range(height):
            for j in range(width):
                depth = gpu_depth_normalized.download()[i, j]  # Download for pixel-level operation
                shift = int((255 - depth) / 255.0 * width / 20)
                right_frame[i, j + shift] = left_frame[i, j]

        # Combine the left and right frames side by side
        stereo_frame = cv2.cuda.hconcat([left_frame, right_frame])

        # Write the stereo frame to the output video from GPU to CPU
        out.write(stereo_frame.download())

    # Release the video objects
    video.release()
    depth_video.release()
    out.release()

argparser = argparse.ArgumentParser(description='Generate 3D video from 2D video and depth map video')

argparser.add_argument('--input', help='Path to input 2D video')
argparser.add_argument('--depth', help='Path to depth map video')
argparser.add_argument('--output', help='Path to output 3D video')

args = argparser.parse_args()

convert_to_stereo_3d(args.input, args.depth, args.output)