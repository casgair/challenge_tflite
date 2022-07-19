"""
Code for generating input images for C++ code, as well as images used for creating and testing tflite model.
Please run in Silent-Face-Anti-Spoofing folder.
"""

import os
import cv2
from src.anti_spoof_predict import Detection
from src.utility import parse_model_name
from src.generate_patches import CropImage


# paths

SAMPLE_IMAGE_PATH = "./images/sample/"
OUTPUT_PATH = "../test/fixtures"
IMAGE_NAMES = [
    "image_F1",
    "image_F2",
    "image_T1",
]

# Load object detection model
model_detection = Detection()
image_cropper = CropImage()

for image_name in IMAGE_NAMES:

    # Load image
    image = cv2.imread(SAMPLE_IMAGE_PATH + image_name + ".jpg")

    # Get bbox for image
    image_bbox = model_detection.get_bbox(image)

    # Crop image
    h_input, w_input, model_type, scale = parse_model_name("2.7_80x80_MiniFASNetV2.pth")
    param = {
        "org_img": image,
        "bbox": image_bbox,
        "scale": scale,
        "out_w": w_input,
        "out_h": h_input,
        "crop": True,
    }
    img = image_cropper.crop(**param)

    # Save image
    image_name_bmp = image_name + ".bmp"
    cv2.imwrite(os.path.join(OUTPUT_PATH, image_name_bmp), img)

print("Images generated in ../test/fixtures")
