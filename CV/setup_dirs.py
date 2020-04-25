import os
import random
import shutil

import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms 

random.seed(0)

labels = [file for file in os.listdir() if file.endswith('.txt')]

label_dict = {}
for file in labels:
  with open(file, 'r') as f:
    for line in f:
      a, b = line.split(' ')[:2]
      b = b.strip('\n')
      if b not in label_dict:
        label_dict[b] = [a]
      else:
        label_dict[b].append(a)

for im in os.listdir('./results/'):
  for k, v in label_dict.items():
    if im[:4] in v and im not in v:
      label_dict[k].append(im[:-4])

images_dict = {'train':{}, 'val':{}}

for k, v in label_dict.items():
  val_images = random.sample(v, int(len(v)*0.2))[:10]
  train_images = list(set(v) - set(val_images))
  num_train = 40 if len(train_images) > 40 else len(train_images)
  train_images = random.sample(train_images, num_train)
  images_dict['val'][k] = val_images
  images_dict['train'][k] = train_images

if not os.path.exists('./images/'):
  os.mkdir('./images/')

if not os.path.exists('./images/train/'):
  os.mkdir('./images/train/')

if not os.path.exists('./images/val/'):
  os.mkdir('./images/val/')

for k, v in images_dict['train'].items():
  os.mkdir(f'./images/train/{k}')
  for pic in v:
      shutil.copy(f'./results/{pic}.JPG', f'./images/train/{k}')

for k, v in images_dict['val'].items():
  os.mkdir(f'./images/val/{k}')
  for pic in v:
      shutil.copy(f'./results/{pic}.JPG', f'./images/val/{k}')
