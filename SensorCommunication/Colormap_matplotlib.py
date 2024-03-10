import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os 

"""
Author: Zishuang Xing

Description:
This script contains a function for applying the 'Inferno' colormap to an image, displaying it with a color bar, 
and saving the processed image. The script is designed to iterate through a series of images in a specified 
directory, apply the colormap, and save the new images in the same directory with a modified name.

"""

def apply_inferno_colormap(image_path, save_path):
    """
    Applies the 'Inferno' colormap to an image, displays it with a color bar, and saves the processed image.

    Parameters:
    image_path (str): The file path of the image to which the colormap will be applied.
    save_path (str): The file path where the processed image will be saved.
    """
    try:
        # Read the image from the provided path
        image = mpimg.imread(image_path)
        
        # Create a figure and axis for the image
        fig, ax = plt.subplots()
        # Apply the Inferno colormap
        cax = ax.imshow(image, cmap='inferno')
        # Add a color bar on the side showing the scale
        fig.colorbar(cax)
        # Save the figure with the image
        plt.savefig(save_path)
        # Close the figure to free memory
        plt.close(fig)
        
        print(f"Image saved to: {save_path}")
    
    except FileNotFoundError:
        # If the file is not found, inform the user
        print(f"File not found: {image_path}")
    except Exception as e:
        # For any other exceptions, print the error
        print(f"An error occurred while processing the image: {e}")

# Base directory and file pattern
base_dir = 'SensorCommunication/Acquisition/20240310-132157'
file_pattern = 'Acquisition-{}.jpg'
save_pattern = 'inferno_{}.jpg' 

# Loop through the range of images
for i in range(20):  # There are images from Acquisition-0 to Acquisition-19
    # Format the file path with the current index
    file_path = os.path.join(base_dir, file_pattern.format(i))
    # Format the save path with the new naming scheme
    save_path = os.path.join(base_dir, save_pattern.format(i))  # inferno_0 for Acquisition-0 and so on
    # Apply the Inferno colormap to the current image and save it
    apply_inferno_colormap(file_path, save_path)
