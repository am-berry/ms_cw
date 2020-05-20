import os
import matplotlib.pyplot as plt

labels = [file for file in os.listdir() if file.startswith('results_') and file.endswith('.txt')]

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

labels = []
lengths = []
for k, v in label_dict.items():
  labels.append(k)
  lengths.append(len(v))

plt.bar(labels, lengths)
plt.xlabel('Label')
plt.ylabel('Number of images')
plt.show()
