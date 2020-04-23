#!/usr/bin/env python3

import cv2
import numpy as np

# Function to extract first frame from video

def extract_frame(vid, output_name):
    vid = cv2.VideoCapture(vid)
    success, image = vid.read()
    if success:
        cv2.imwrite(f"{output_name}.JPG", image)


