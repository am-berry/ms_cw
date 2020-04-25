from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os

if __name__ == '__main__':
  data_dir = './images/'
  batch_size =16 
  epochs = 20
  workers = 8

  #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  device = 'cpu'
  print(f'Running on device: {device}')

  resnet = InceptionResnetV1(
      classify=True,
      pretrained='vggface2',
      num_classes=48
  ).to(device)

  optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
  scheduler = MultiStepLR(optimizer, [5, 10])

  data_transforms = {
      'train': transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }

  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
  dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                               shuffle=True, num_workers=0)
                for x in ['train', 'val']}
  dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
  class_names = image_datasets['train'].classes

  loss_fn = torch.nn.CrossEntropyLoss()
  metrics = {
      'fps': training.BatchTimer(),
      'acc': training.accuracy
  }

  writer = SummaryWriter()
  writer.iteration, writer.interval = 0, 10

  print('\n\nInitial')
  print('-' * 10)
  resnet = resnet.eval()
  training.pass_epoch(
      resnet, loss_fn, dataloaders['val'],
      batch_metrics=metrics, show_running=True, device=device,
      writer=writer
  )

  for epoch in range(epochs):
      print('\nEpoch {}/{}'.format(epoch + 1, epochs))
      print('-' * 10)

      resnet = resnet.train()
      training.pass_epoch(
          resnet, loss_fn, dataloaders['train'], optimizer, scheduler,
          batch_metrics=metrics, show_running=True, device=device,
          writer = writer
      )
      resnet = resnet.eval()
      training.pass_epoch(
          resnet, loss_fn, dataloaders['val'],
          batch_metrics=metrics, show_running=True, device=device,
          writer=writer
      )

  writer.close()

  torch.save(resnet.state_dict(), "./models/resnet_retrained.pth")


