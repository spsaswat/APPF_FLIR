import time
import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime
import os
import json

# Initialize the global variable
current_position = 0.0
all_imgs = []

# serial_number = 'f1230450' #L515
# serial_number = '017322071325' #D435
serial_number = '128422272123'  #D405

save_fold_p = './data/test_plant_'

now = datetime.now()
dt_string = now.strftime("%Y%m%d%H%M%S")
save_fold_p = save_fold_p + dt_string + '/'

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


def capture_images(pipeline, total_images=20, delay=1):
    for i in range(total_images):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        position_str = f"{i+1:06d}"  # Adjust the file naming as per requirements
        
        cv2.imwrite(os.path.join(save_fold_p, f'rgb_{position_str}.png'), color_image)
        cv2.imwrite(os.path.join(save_fold_p, f'depth_{position_str}.png'), depth_image)

        time.sleep(delay)

# Configure the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(serial_number)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # Adjust as per your device's capabilities
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)    # Adjust as per your device's capabilities
align = rs.align(rs.stream.color)

start_pipeline()
save_intrinsics()
capture_images(pipeline, 20, 1)
stop_pipeline()

