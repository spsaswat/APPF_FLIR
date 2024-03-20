import time
import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize the global variable
current_position = 0.0
all_imgs = []

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


def capture_images(pipeline, position):
    # The pipeline is assumed to already be started
    # time_0 = time.time()

    # Capture one frame
    frames = pipeline.wait_for_frames()

    # time_1 = time.time()
    # print((time_1 - time_0)*1000)


    aligned_frames = align.process(frames)

    # Get aligned depth and color frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # time_2 = time.time()
    # print((time_2 - time_0)*1000)

    # Convert depth and color frames to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # time_3 = time.time()
    # print((time_3 - time_0)*1000)

    # Save images
    position = int(position * 10**6)
    
#     Change accrding to camera requirements
    all_imgs.append((color_image.copy(), save_fold_p+'rgb_'+str(position)+'.png'))
    # cv2.imwrite(save_fold_p+'rgb_'+str(position)+'.png' , color_image)
    cv2.imwrite(save_fold_p+'depth_'+str(position)+'.png', depth_image)
    # all_imgs.append((depth_image.copy(), save_fold_p+'depth_'+str(position)+'.png'))

    # time_4 = time.time()
    # print((time_4 - time_0)*1000)

    

    # Save images
    cv2.imwrite(f'./data/test_plant/rgb_{position}.jpeg', color_image)
    cv2.imwrite(f'./data/test_plant/depth_{position}.jpeg', depth_image)
