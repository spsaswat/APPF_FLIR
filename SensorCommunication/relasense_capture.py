import time
import pyrealsense2 as rs
import numpy as np
import cv2

def capture_images(position):
    
    # Set the serial number of the camera you want to use
    serial_number = 'f1230450'

    # Configure the RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    
    # Start streaming
    pipeline.start(config)

    # Capture one frame
    frames = pipeline.wait_for_frames()

    # Align depth frame to color frame
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)

    # Get aligned depth and color frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # Convert depth and color frames to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Stop streaming
    pipeline.stop()
    

    # Save images
    cv2.imwrite(f'./data/test_plant/rgb_{position}.jpeg', color_image)
    cv2.imwrite(f'./data/test_plant/depth_{position}.jpeg', depth_image)
