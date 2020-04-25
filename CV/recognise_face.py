import face_detector

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import PIL

from retinaface.models.retinaface import RetinaFace
from retinaface.utils.box_utils import decode
from retinaface.utils.timer import Timer
from retinaface.data import cfg_re50
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.utils.nms.py_cpu_nms import py_cpu_nms

from retinaface_utils import * 

import numpy as np
import cv2

import joblib

import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

class_names = os.listdir('./images/train/')

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
    model = models.resnet50(pretrained=False)
    num_f = model.fc.in_features
    model.fc = nn.Linear(num_f, 48)
    model.load_state_dict(torch.load("Resnet50_retrained.pth", map_location=device))
    model.to(device)
    model.eval()
    data_transforms = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
  tensor = torch.Tensor(1,3,224,224)
  centres = []
  results = {}
  idx = []
  for face in faces:
    centre = (int((face[2]-face[0]) / 2), int((face[3]-face[1]) / 2))
    centres.append(centre)
    face_dict[centre] = img[int(face[1]):int(face[3]), int(face[0]):int(face[2])]
    face = PIL.Image.fromarray(face_dict[centre])
    face_tensor = data_transforms(face).float()
    face_tensor = face_tensor.unsqueeze_(0)
    inp = Variable(face_tensor).to(device)
    out = model(inp)
    index = out.data.cpu().numpy().argmax()
    results[index] = centre
    idx.append(index)
  return face_dict, results, centres, idx
  
if __name__ == '__main__':
    face, results, centres, idx = recognise_face('IMG_6851.JPG', classifier_type = 'cnn')
    im = cv2.imread('IMG_6851.JPG')
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k, v in results.items():
      print(class_names[k])
