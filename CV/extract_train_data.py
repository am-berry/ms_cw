#!/usr/bin/env python3
# This script extracts the frames from each 
import os

import numpy as np
import cv2

def extract_frames(vid, save_path):
  if not os.path.exists(save_path):
    os.mkdir(save_path)
  vidcap = cv2.VideoCapture(vid)
  count = 0
  save_name = vid[:-4]
  while True:
    success, frame = vidcap.read()
    if not success:
      break
    save = save_name + "_" + str(count) + ".JPG"
    cv2.imwrite(save_path + save, frame)
    count += 1


if __name__ == "__main__":

  vid_list = [file for file in os.listdir() if 'IndividualVideos' in file]
  for folder in vid_list:
    print(f"Extracting frames from {folder}")
    for vid in os.listdir(f"./{folder}/"):
      extract_frames(f"./{folder}/{vid}", "./IndividualImages6/")
      print(f"frames extracted from {vid}!")
  print("Complete")
