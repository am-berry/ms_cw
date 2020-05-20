import os

for pic in os.listdir():
  if "augmented" in pic:
    os.remove(pic)
