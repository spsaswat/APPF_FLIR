import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

"""
Author: Zishuang Xing, Yash Srivastava

Description:
This script contains functions for applying the 'Inferno' colormap and a custom colormap to an image, displaying it with
a color bar, and saving the processed image. The script is designed to iterate through a series of images in a specified 
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


def apply_custom_colormap(image_path, save_path):
    """
    Applies a custom colormap to the images, displays it with a color bar, and saves the processed image.

    Parameters:
    image_path (str): The file path of the image to which the colormap will be applied.
    save_path (str): The file path where the processed image will be saved.
    """
     # Load the image
    image = mpimg.imread(image_path)

    # Define the custom colormap
    # [       Black,    Dark Purple,    Blue,       Cyan,     Green,    Yellow,     Orange,       Red,      White]
    cmap = [(0, 0, 0), (0.5, 0, 0.5), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (1, 1, 1)]

    # Convert to LinearSegmented Colormap
    custom_colormap = LinearSegmentedColormap.from_list("custom_ironbow", cmap, N=256)

    # Create a figure and axis for the image
    fig, ax = plt.subplots()
    # Apply the custom colormap
    cax = ax.imshow(image, cmap=custom_colormap)
    # Add a color bar on the side showing the scale
    fig.colorbar(cax)
    # Save the figure with the image
    plt.savefig(save_path)
    # Close the figure to free memory
    plt.close(fig)
    
    print(f"Image saved to: {save_path}")

def process_directory(directory, colormap_choice):
    """
    Processes all images in the specified directory using the selected colormap.

    Parameters:
    directory (str): The directory containing the images.
    colormap_choice (str): The selected colormap ('inferno' or 'custom').
    """
    
    #Get a list of all files in the directory
    files = os.listdir(directory)
    #Compile regular expression patterns to match file names in the format of "on number. jpg
    pattern = re.compile(r'on-(\d+).jpg$')
    #Iteration of the file list, retaining only the file names that match the regular expression
    files_with_numbers = [(f, int(pattern.search(f).group(1))) for f in files if pattern.search(f)]
    #Sort based on the numerical part of the file name
    files_with_numbers.sort(key=lambda x: x[1])

    #Colour processing each files
    for file_name, number in files_with_numbers:
        #The complete path to the construction file
        file_path = os.path.join(directory, file_name)
        
        # Construct a save path for processed images based on the colormap choice
        if colormap_choice.lower() == 'inferno':
            # For inferno colormap
            save_pattern = 'inferno_{}.jpg'
            save_path = os.path.join(directory, save_pattern.format(number))
        elif colormap_choice.lower() == 'custom':
            save_pattern = 'c_ironbow_{}.jpg'
            # For custom colormap
            save_path = os.path.join(directory, save_pattern.format(number))
        else:
            print(f"Invalid colormap choice: {colormap_choice}")
            return
        
        #Process and save images based on user selection
        if colormap_choice.lower() == 'inferno':
            #Execute inferno color processing
            apply_inferno_colormap(file_path, save_path)
        elif colormap_choice.lower() == 'custom':
            #Execute inferno custom processing
            apply_custom_colormap(file_path, save_path)
        else:
            #Report errors
            print(f"Invalid colormap choice: {colormap_choice}")


# Base directory and file pattern
base_dir = 'SensorCommunication/Acquisition/20240310-132157'

# Prompt user for colormap choice
colormap_choice = input("Enter 'inferno' to use the standard colormap or 'custom' to use a custom colormap: ")

# Process each directory
process_directory(base_dir, colormap_choice)

# Wait for the user to press Enter to exit
input("Processing complete. Press Enter to exit...")