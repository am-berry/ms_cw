import os
import joblib

import cv2
import numpy as np

from scipy.cluster.vq import vq, kmeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# function to extract descriptors using SIFT or SURF
def extract_descriptors(train_dir, extractor):
  if extractor.lower() not in ['sift', 'surf']:
    raise ValueError("extractor must be either sift or surf")
  if extractor.lower() == 'surf':
    detector = cv2.xfeatures2d.SURF_create()
  else:
    detector = cv2.xfeatures2d.SIFT_create()
  desc = {}
  labels = []
  for dir in os.listdir(train_dir):
    label = dir
    for im in os.listdir(f'{train_dir}/{dir}'):
      img = cv2.imread(f'{train_dir}/{dir}/{im}')
      img = cv2.resize(img, (224,224))
      kp, des = detector.detectAndCompute(img, None)
      desc[im] = des
      labels.append(label)
  return desc, labels 

# stacks arrays vertically
def stack_array(desc):
  desc = list(desc.values())
  descriptors = np.array(desc[0])
  for descriptor in desc[1:]:
    descriptors = np.vstack((descriptors, descriptor))
  return descriptors

# K-means clustering
def perform_kmeans(descriptors, cluster_size):
  vocabulary, _ = kmeans(descriptors, cluster_size, 1)
  return vocabulary

# Returns codebook i.e. bag of visual words via vector quantization
def extract_features(unstacked_descriptors, size, vocabulary):
  desc = list(unstacked_descriptors.values())
  im_features = np.zeros((len(desc), size), 'float32')
  for i in range(len(desc)):
    words, distance = vq(desc[i], vocabulary)
    for w in words:
      im_features[i][w] += 1
  return im_features
