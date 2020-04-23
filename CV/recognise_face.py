#!/usr/bin/env python3

import face_detector

import os
import torch
import torch.backends.cudnn as cudnn

from retinaface.models.retinaface import RetinaFace
from retinaface.utils.box_utils import decode
from retinaface.utils.timer import Timer
from retinaface.data import cfg_re50
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.utils.nms.py_cpu_nms import py_cpu_nms

from retinaface_utils import * 

import numpy as np
import cv2

import recogniser

import joblib

import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

def recognise_face(image, feature_type=False, classifier_type=False, creative_mode=0):
  # load retinaface model 
  net = RetinaFace(cfg=cfg_re50, phase = 'test')
  net = load_model(net, 'Resnet50_Final.pth', False)
  net.eval()
  print('Model loaded successfully')
  cudnn.benchmark = True
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net = net.to(device)

  # detect faces in image
  faces = face_detector.face_detector(image, out_name = 'Results', net = net, save_image=True)
  
  img = cv2.imread(image, cv2.IMREAD_COLOR)
  face_dict = {}
  if classifier_type == 'cnn':
      model = recogniser.Model()
      model.load_state_dict(torch.load(...))
  if classifier_type == 'svm' and feature_type == 'hog':
      model = joblib.load('svm_hog.joblib')
  if classifier_type == 'svm' and feature_type == 'sift':
      model = joblib.load('svm_sift.joblib')
  if classifier_type == 'svm' and feature_type
 
 for face in faces:
    centre = ((face[2]-face[0]) / 2, (face[3]-face[1]) / 2)
    face_dict[centre] = img[int(face[1]):int(face[3]), int(face[0]):int(face[2])]
    cv2.imwrite(f'img{centre[0]}_{centre[1]}.JPG', face_dict[centre])
    
  return face_dict 
  
if __name__ == '__main__':
    print(recognise_face('IMG_6831.JPG'))
