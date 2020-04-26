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


def recognise_face_cnn(detected_dir):
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
  classes = datasets.ImageFolder('./images/train/').classes 
  labels = {}
  proba = []
  for im in os.listdir(detected_dir):
    lab, img, probs = cnn_inference(f'{detected_dir}/{im}', model, classes, data_transforms)
    labels[im] = list(probs)
    proba.append(list(probs))

  proba.sort(key=max, reverse=True)
  ord = []
  argmaxes = []
  probs = []
  for prob in proba:
    for k, v in labels.items():
      if p == v:
        ord.append(k)
    n = 0
    sorted_ps = sorted(prob, reverse=True)
    while prob.index(sorted_ps[n]) in argmaxes:
      n+=1
    argmaxes.append(prob.index(sorted_ps[n]))
    probs.append(sorted_ps[n])
  labs = []
  for i in argmaxes:
    labs.append(classes[i])
  fin = dict(zip(ord, labs))
  return fin, probs 

def cnn_inference(img, model, labels, data_transforms):
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  im = PIL.Image.open(img)
  face_tensor = data_transforms(im).float().unsqueeze_(0)
  inp = Variable(face_tensor).to(device)
  out = model(inp)
  probs = nn.functional.softmax(out).cpu().detach().numpy().flatten()
  index = out.data.cpu().numpy().argmax()
  return labels[index], img, probs

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
  centres = []
  results = {}
  for face in faces:
    centre = (int((face[2]+face[0]) / 2), int((face[3]+face[1]) / 2))
    centres.append(centre)
    face_dict[centre] = img[int(face[1]):int(face[3]), int(face[0]):int(face[2])]
    cv2.imwrite(f'./det_faces/face_{centre[0]}_{centre[1]}.PNG', face_dict[centre])

  if classifier_type == 'cnn':
    fin, prob = recognise_face_cnn('./det_faces/')
     
  return fin, prob 

if __name__ == '__main__':
  im = 'IMG_6854.JPG'
  fin, prob = recognise_face(im, classifier_type = 'cnn')
  im = cv2.imread(im)
  centres = [tuple(x.strip('.JPG').strip('.PNG').split('_')[1:3]) for x in fin.keys()]
  for x,y,z in zip(centres, fin.values(), prob):
    print(f'{x} - {y} - {z}')
    im = cv2.putText(im, f'{y} - Probability {z:.3f}', (int(x[0]), int(x[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
  cv2.imshow('img', im)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.imwrite('results.JPG', im)
