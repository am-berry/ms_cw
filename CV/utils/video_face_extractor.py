import os
import torch
import torch.backends.cudnn as cudnn
import face_detector
from retinaface.models.retinaface import RetinaFace
from retinaface.utils.box_utils import decode
from retinaface.utils.timer import Timer
from retinaface.data import cfg_re50
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.utils.nms.py_cpu_nms import py_cpu_nms

from retinaface_utils import * 

import numpy as np
import cv2

def extract_faces_video(vid_path):
  cap = cv2.VideoCapture(vid_path)
  vid = vid_path[-8:-4]
  success, img = cap.read()
  count = 0
  while success:
    success, img = cap.read()
    if count % 10 == 0:
      cv2.imwrite(f'./vid_reduced/{vid}_{count}.JPG', img)
    count += 1

if __name__ == "__main__":
#  vid_folders = [file for file in os.listdir() if "IndividualVideos" in file]
#  for folder in vid_folders:
#    for vid in os.listdir(f'./{folder}/'):
#      extract_faces_video(f'./{folder}/{vid}')
  
  net = RetinaFace(cfg = cfg_re50, phase = 'test')
  net = load_model(net, 'Resnet50_Final.pth', False)
  net.eval()
  print('Model loaded successfully')
  cudnn.benchmark = True
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net = net.to(device)

  for pic in os.listdir('./vid_reduced/'):
    faces = face_detector.face_detector(f'./vid_reduced/{pic}', net=net, save_image=False)
    im = cv2.imread(f'./vid_reduced/{pic}', cv2.IMREAD_COLOR)
    for face in faces:
      new = im[int(face[1]):int(face[3]), int(face[0]):int(face[2])] 
      cv2.imwrite(f'./results/{pic}', new)
