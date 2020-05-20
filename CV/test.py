import os
import cv2
import time

import numpy as np
import recognise_face 

# testing on 10 images, over all parameters

images = os.listdir('./test_images/')
extractors = ['sift', 'surf']
models = ['svm', 'rf']

for im in images:
  for model in models:
      for extractor in extractors:
        fin, _ = recognise_face.recognise_face(os.path.join('./test_images', im), feature_type = extractor, classifier_type = model, creative_mode = 0)

        img = cv2.imread(f'./test_images/{im}')
        centres = [tuple(x.strip('.JPG').strip('.PNG').split('_')[1:3]) for x in fin.keys()]
        for x, y in zip(centres, fin.values()):
          print(f'{y} {x[0]} {x[1]}')
          img = cv2.putText(img, f'{y}', (int(x[0]), int(x[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        t = str(time.time()).split('.')[0]
        cv2.imwrite(f'{extractor}_{model}_results{t}.JPG', img)
