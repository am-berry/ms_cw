import os
from PIL import Image

x = ['train', 'val']

for s in x:
  for folder in os.listdir(f'./images/{s}/'):
    for im in os.listdir(f'./images/{s}/{folder}/'):
      try:
        img = Image.open(f'./images/{s}/{folder}/{im}')
      except:
        print(f'File removed - {im}')
        os.remove(f'./images/{s}/{folder}/{im}')
