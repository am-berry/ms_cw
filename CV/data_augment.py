import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_im(img):
  return cv2.resize(img, (80,80))

def reduce_brightness(img, val=0.8):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  hsv[...,2] = hsv[...,2]*val
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def gaussian_blur(img, kernel = (5,5)):
  return cv2.GaussianBlur(img, kernel, 0)

if __name__ == "__main__":
  for pic in os.listdir('./results/'):
    img = cv2.imread(f'./results/{pic}')
    resized = resize_im(img)
    dull = reduce_brightness(resized, val = 0.7)
    blur = gaussian_blur(dull, kernel = (5,5))
    cv2.imwrite(f'./results/{pic[:-4]}_augmented.JPG', blur)
