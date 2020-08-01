import os
import argparse
import glob 
import time
import joblib
import warnings

import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

from scipy.cluster.vq import vq
import numpy as np
import cv2
import PIL

import face_detector
import creative
from retinaface.models.retinaface import RetinaFace
from retinaface.data import cfg_re50
from retinaface_utils import * 
import feature_extractors

import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

warnings.filterwarnings('ignore')
class_names =['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', 
'16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', 
'33', '34', '36', '38', '40', '42', '44', '46', '48', '50', '52', '54', '56', '58', '60', '78']
 
def recognise_face_cnn(detected_dir, class_names):
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  model = models.resnet50(pretrained=False)
  num_f = model.fc.in_features
  model.fc = nn.Linear(num_f, 48)
  model.load_state_dict(torch.load("./models/Resnet50_retrained.pth", map_location=device))
  model.to(device)
  model.eval()
  data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  labels = {}
  proba = []
  for im in os.listdir(detected_dir):
    lab, img, probs = cnn_inference(f'{detected_dir}/{im}', model, class_names, data_transforms)
    labels[im] = list(probs)
    proba.append(list(probs))

  proba.sort(key=max, reverse=True)
  ord = []
  argmaxes = []
  probs = []
  for prob in proba:
    for k, v in labels.items():
      if prob == v:
        ord.append(k)
    n = 0
    sorted_ps = sorted(prob, reverse=True)
    while prob.index(sorted_ps[n]) in argmaxes:
      n+=1
    argmaxes.append(prob.index(sorted_ps[n]))
    probs.append(sorted_ps[n])
  labs = []
  for i in argmaxes:
    labs.append(class_names[i])
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

def recognise_face_classic(detected_dir, classifier, extractor, detector, scaler, vocab):
  if extractor == "sift" and classifier == 'svm':
    model = joblib.load("./models/SVM_SIFT.joblib")
  if extractor == "surf" and classifier == 'svm':
    model = joblib.load("./models/SVM_SURF.joblib")

  if extractor == "sift" and classifier == 'rf':
    model = joblib.load("./models/RF_SIFT.joblib")
  if extractor == "surf" and classifier == 'rf':
    model = joblib.load("./models/RF_SURF.joblib")

  outs = {}
  for im in os.listdir(detected_dir):
    img = cv2.imread(f'{detected_dir}/{im}')
    img = cv2.resize(img, (224,224))
    kp, des = detector.detectAndCompute(img, None)
    im_features = np.zeros(800, 'float32')
    words, distance = vq(des, vocab)
    for w in words:
      im_features[w] += 1
    bovw = scaler.transform(im_features.reshape(1,-1))
    preds = model.predict(bovw)[0]
    outs[im] = preds
  return outs

def recognise_face(image, feature_type='sift', classifier_type='svm', creative_mode=0):
  class_names =['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', 
  '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', 
  '33', '34', '36', '38', '40', '42', '44', '46', '48', '50', '52', '54', '56', '58', '60', '78']
  # load retinaface model 
  net = RetinaFace(cfg=cfg_re50, phase = 'test')
  net = load_model(net, './models/Resnet50_Final.pth', False)
  net.eval()
  print('Model loaded successfully')
  cudnn.benchmark = True
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net = net.to(device)

  # detect faces in image
  faces = face_detector.face_detector(image, out_name = 'results', net = net, save_image=True)
  
  img = cv2.imread(image, cv2.IMREAD_COLOR)

  centres = []
  bboxes = []
  results = {}
  if not os.path.exists("./det_faces/"):
    os.mkdir("./det_faces/")
  else:
    files = glob.glob("./det_faces/*")
    for f in files:
      os.remove(f)
  for face in faces:
    centre = (int((face[2]+face[0]) / 2), int((face[3]+face[1]) / 2))
    centres.append(centre)
    image = img[int(face[1]):int(face[3]), int(face[0]):int(face[2])]
    cv2.imwrite(f'./det_faces/face_{centre[0]}_{centre[1]}.PNG', image)
    bbox = [int(f) for f in face[0:4]]
    bboxes.append(bbox)
  if creative_mode:
    whole = cv2.imread('./results.jpg', cv2.IMREAD_COLOR)
    for face in bboxes:
      roi = whole[face[1]:face[3], face[0]:face[2],:]
      cartoon = creative.cartoonify(roi)
      whole[face[1]:face[3], face[0]:face[2], :] = cartoon
    cv2.imwrite('results.jpg', whole)
  if feature_type == 'sift':
    scaler = joblib.load('./models/SIFT_SCALER.bin')
    vocab = np.load('./models/sift_vocab.npy')
    detector = cv2.xfeatures2d.SIFT_create()
  if feature_type == 'surf':
    scaler = joblib.load('./models/SURF_SCALER.bin')
    vocab = np.load('./models/surf_vocab.npy')
    detector = cv2.xfeatures2d.SURF_create()
  if classifier_type.lower() == 'cnn':
    fin, prob = recognise_face_cnn('./det_faces/', class_names)
  elif classifier_type.lower() == 'svm' or classifier_type.lower() == 'rf':
    fin =  recognise_face_classic('./det_faces/', classifier = classifier_type, extractor = feature_type, detector = detector, scaler = scaler, vocab = vocab)
    prob = None
  return fin, prob 

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('image', type = str, help = 'image filename with appropriate path "./*.JPG/PNG')
  parser.add_argument('classifier', type = str, help = 'type of classifier, CNN, SVM, RF')
  parser.add_argument('--features', type = str, help = 'type of feature extractor used, only with SVM and RF - either surf or sift')
  parser.add_argument('--creative_mode', default = False, action = 'store_true', help = "set 1 to add cartoonifying to faces in group image")
  args = parser.parse_args()
  fin, _ = recognise_face(args.image, classifier_type = args.classifier, feature_type = args.features, creative_mode = args.creative_mode)

  #load in copy of image with bounding boxes added
  im = cv2.imread('results.jpg', cv2.IMREAD_COLOR)
  centres = [tuple(x.strip('.JPG').strip('.PNG').split('_')[1:3]) for x in fin.keys()]
  if os.path.exists('./results.txt'):
    os.remove('./results.txt')
  with open('results.txt', 'w') as f:
    for x,y in zip(centres, fin.values()):
      print(f'{y} {x[0]} {x[1]}')
      im = cv2.putText(im, f'{y}', (int(x[0]), int(x[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
      f.write(f'{y} - {x[0]}, {x[1]}\n') 
  f.close()
  t = str(time.time()).split('.')[0]
  os.remove('results.JPG') 
  cv2.imwrite(f'results{t}.JPG', im)
