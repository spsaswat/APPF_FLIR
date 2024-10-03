import cv2
import numpy as np

# Load the RGB image
image = cv2.imread('Acquisition/batch_1/test_plant_20241003135024/rgb_1340.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Scale the image to 16-bit
scaled = (gray.astype(np.float32) / 255) * 65535
scaled = scaled.astype(np.uint16)

# Function to get pixel value on mouse click
def get_pixel_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = scaled[y, x]
        print(f"Pixel value at ({x}, {y}): {pixel_value}")

# Display the image and set mouse callback
cv2.imshow('Thermal Image', scaled)

# Set the mouse callback
cv2.setMouseCallback('Thermal Image', get_pixel_value)

# Wait until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
