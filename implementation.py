# import islice to simplify iteration
from itertools import islice

# import the ability to make shapes through Pillow
from draw import DrawAShape
from PIL import Image

# import the UNet
from unet import UNet

# import torch and numpy
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torchvision

# import time to keep track of it and os to create directories
import time
import os

# create directories to save the predictions and ground truth
if not os.path.exists('./predictions'):
  os.makedirs('./predictions')
if not os.path.exists('./ground_truth'):
  os.makedirs('./ground_truth')

# clear the directory of any previous images
for file in os.listdir('./predictions'):
  os.remove(f'./predictions/{file}')
for file in os.listdir('./ground_truth'):
  os.remove(f'./ground_truth/{file}')

start_time = time.time()

class RandomShapesDataset(torch.utils.data.IterableDataset):
  class RandomShapesIterator:
    def __init__(self):
      # transform the images to tensors so they can be used in the model
      self.transform = transforms.Compose([
          transforms.ToTensor(),
          # convert the images to grayscale to ensure they are 1 channel
          transforms.Grayscale(num_output_channels=1)
      ])

    def __next__(self):
      DrawAShape()
      outline = Image.open('outline.jpg')
      filled = Image.open('filled.jpg')

      # convert the images to tensors
      outline = self.transform(outline)
      filled = self.transform(filled)
      
      return outline, filled
    
  def __init__(self):
    super().__init__()

  def __iter__(self):
    return RandomShapesDataset.RandomShapesIterator()

# initialize the dataset
train_dataset = RandomShapesDataset()

# load the dataset into a loader with a small batch size for now
train_loader = DataLoader(train_dataset, batch_size=4)

# define model
model = UNet(n_class=1)

# make sure the model is on the GPU if it is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_iterations = 0

# iterate for 1000 times over 4000 total images (1000 x batch size of 4)
for x_batch, y_batch in islice(train_loader, 5000):
  x_batch, y_batch = x_batch.to(device), y_batch.to(device)
  
  model_predictions = model(x_batch)
  loss = F.binary_cross_entropy_with_logits(model_predictions, y_batch)

  # use torchvision.transforms.ToPILImage(mode=None) to convert the tensor to a PIL Image
  # then use the save method to save the image
  # only save every 50 iterations
  if num_iterations % 50 == 0:
    # use the sigmoid function to convert the predictions to a probability
    image_prediction = torch.sigmoid(model_predictions[0])
    torchvision.transforms.ToPILImage(mode=None)(image_prediction[0].squeeze()).save(f'./predictions/p_{num_iterations}.jpg')
    torchvision.transforms.ToPILImage(mode=None)(y_batch[0].squeeze()).save(f'./ground_truth/g_{num_iterations}.jpg')

  # backpropagation
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # calculate the elapsed time and format it
  elapsed_time = time.time() - start_time
  formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

  print(f'Iteration: {num_iterations} | Time: {formatted_time} | Loss: {loss.item()}')
  num_iterations += 1