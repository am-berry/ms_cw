import os

import cv2
import numpy as np

def creative_mode(ind_imgs):
  for im in ind_imgs:
    cartoonify(im)

def cartoonify(img):
  img_new = img / 255.0
  img_new = img_new.reshape((-1,3)).astype(np.float32)
  flags = cv2.KMEANS_RANDOM_CENTERS
  compactness, labels, centres = cv2.kmeans(img_new, 16, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, flags)
  new_colours = centres[labels].reshape((-1,3))
  cartoon = new_colours.reshape(img.shape)
  return cartoon * 255.0

if __name__ == '__main__':
  img = cv2.imread('roi.JPG', cv2.IMREAD_COLOR)
  cartoon = cartoonify(img)
  print(cartoon)
  print(cartoon.shape)
  cv2.imwrite('cartoon.JPG', cartoon)
  cv2.imshow('cartoon', cartoon)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
