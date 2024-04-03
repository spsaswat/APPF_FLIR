import time
import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime
import os
import json

"""
Author: spsaswat

Description:
This script demonstrates the process of acquiring images using the Intel RealSense camera via the pyrealsense2 library. 
The script showcases essential steps including starting and stopping the camera pipeline, capturing a specified number 
of images with a delay between each capture, saving camera intrinsics, and handling file paths dynamically based on the 
execution time. This utility is particularly useful in computer vision applications that require synchronized color and 
depth data from RealSense cameras.

"""

# Initialize the global variable
current_position = 0.0
all_imgs = []

# serial_number = 'f1230450' #L515
# serial_number = '017322071325' #D435
serial_number = '128422272123'  #D405

#Set the basic path for data storage
save_fold_p = './data/test_plant_'

now = datetime.now()
dt_string = now.strftime("%Y%m%d%H%M%S")
save_fold_p = save_fold_p + dt_string + '/'
#If it does not exist, create a save directory
os.makedirs(save_fold_p, exist_ok=True) 

def start_pipeline():
    # Start streaming
    pipeline.start(config)
    # Get the depth sensor and set the visual preset


def stop_pipeline():
    # Stop streaming
    pipeline.stop()
    

def save_intrinsics():
    # Get the intrinsics
    profile = pipeline.get_active_profile()
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_sensor.set_option(rs.option.visual_preset, 4)  # 4 High accuracy for D435 and D405 # 5 L515 short range 

    # depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))  #change it back if it doesnot work
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()

    # Save the intrinsics in the required format
    intrinsics_dict = {
        "K": [
            [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
            [0, depth_intrinsics.fy, depth_intrinsics.ppy],
            [0, 0, 1]
        ],
        # Assuming that the distortion model is "Brown-Conrady",
        # you can get the distortion parameters like this.
        "dist": depth_intrinsics.coeffs,
        "height": depth_intrinsics.height,
        "width": depth_intrinsics.width
    }

    # Write the data to a .txt file
    with open(save_fold_p+'kd_intrinsics.txt', 'w') as outfile:
        json.dump(intrinsics_dict, outfile, indent=4)
        
        
    #remove below code after intrinsics finalized
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color)) 
    depth_intrinsics = depth_profile.get_intrinsics()

    # Save the intrinsics in the required format
    intrinsics_dict = {
        "K": [
            [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
            [0, depth_intrinsics.fy, depth_intrinsics.ppy],
            [0, 0, 1]
        ],
        # Assuming that the distortion model is "Brown-Conrady",
        # you can get the distortion parameters like this.
        "dist": depth_intrinsics.coeffs,
        "height": depth_intrinsics.height,
        "width": depth_intrinsics.width
    }

    # Write the data to a .txt file
    with open(save_fold_p+'kdc_intrinsics.txt', 'w') as outfile:
        json.dump(intrinsics_dict, outfile, indent=4)


def capture_images(pipeline, total_images=10, delay=1):
    # Capture a specified number of images, pausing for a set time between captures
    for i in range(total_images):
        frames = pipeline.wait_for_frames()
        # Align depth frames to color frames to ensure depth and color data match
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # Convert frame data to numpy arrays for processing and saving with OpenCV
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Save color and depth images with filenames "rgb_x.png" and "depth_x.png", where x is the sequence number
        position_str = f"{i+1:06d}" 
        
        cv2.imwrite(os.path.join(save_fold_p, f'rgb_{position_str}.png'), color_image)
        cv2.imwrite(os.path.join(save_fold_p, f'depth_{position_str}.png'), depth_image)

        time.sleep(delay) # Wait for a specified time (in seconds) between captures

# Configure the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(serial_number)
# Configure the stream parameters according to device capabilities and needs
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) 
align = rs.align(rs.stream.color) #Align depth frames to color frames

# Sequentially execute starting the pipeline, saving intrinsics, capturing images, and stopping the pipeline
start_pipeline()
save_intrinsics()
capture_images(pipeline, 10, 1)
stop_pipeline()