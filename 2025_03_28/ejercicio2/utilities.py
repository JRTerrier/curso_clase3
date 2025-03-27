import cv2
import numpy as np


def convert_rgb_to_lab_hsv(rgb_triplet):
    # Normalize the RGB values to the range [0, 1] if they are not already in the range [0, 255]
    rgb_array = np.array([[rgb_triplet]], dtype=np.float32)
    if np.max(rgb_array) <= 1.0:
        rgb_array = rgb_array * 255  # Scale to [0, 255]
    
    # Convert the RGB array to uint8
    rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
    
    # Convert from RGB to CIE-Lab
    lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2Lab)
    lab_triplet = lab_array[0, 0].astype(np.float32)
    
    # Scale L to [0, 100] and shift a and b from [0, 255] to [-128, 127]
    lab_triplet[0] = lab_triplet[0] * 100 / 255
    lab_triplet[1] = lab_triplet[1] - 128
    lab_triplet[2] = lab_triplet[2] - 128
    
    # Convert from RGB to HSV
    hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    hsv_triplet = hsv_array[0, 0].astype(np.float32)
    
    # Scale HSV values to the standard ranges
    hsv_triplet[0] = hsv_triplet[0] * 2  # H ranges from 0 to 360 in degrees
    hsv_triplet[1] = hsv_triplet[1] / 255 * 100  # S ranges from 0 to 100 percent
    hsv_triplet[2] = hsv_triplet[2] / 255 * 100  # V ranges from 0 to 100 percent
    
    return lab_triplet, hsv_triplet
