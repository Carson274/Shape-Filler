from itertools import islice
from draw import DrawAShape
from PIL import Image
from old_unet import UNet
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
import time
import os
import matplotlib.pyplot as plt

start_time = time.time()

class RandomShapesDataset(torch.utils.data.IterableDataset):
    class RandomShapesIterator:
        def __init__(self):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1)
            ])

        def __next__(self):
            DrawAShape()
            outline = Image.open('outline.jpg')
            filled = Image.open('filled.jpg')
            outline = self.transform(outline)
            filled = self.transform(filled)
            return outline, filled

    def __init__(self):
        super().__init__()

    def __iter__(self):
        return RandomShapesDataset.RandomShapesIterator()

train_dataset = RandomShapesDataset()
train_loader = DataLoader(train_dataset, batch_size=8)
model = torch.load('models/unet_model.pt')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_iterations = 0
iou_values = []

def calculate_iou(preds, labels):
    # ensure preds is a boolean tensor
    preds_bool = preds.bool()
    labels_bool = labels.bool()

    intersection = (preds_bool & labels_bool).float().sum((1, 2))  # compute intersection
    union = (preds_bool | labels_bool).float().sum((1, 2))         # compute union
    
    iou = (intersection + 1e-6) / (union + 1e-6)    # smooth division to avoid 0/0
    return iou.mean().item()                        # return the mean IoU score for the batch

# train on 800 images (100 iterations x batches of 8)
for x_batch, y_batch in islice(train_loader, 100):
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    with torch.no_grad():
        model_predictions = model(x_batch)
        image_prediction = torch.sigmoid(model_predictions)
        
        # convert probabilities to binary predictions
        preds = image_prediction > 0.5

        # calculate IoU
        iou_score = calculate_iou(preds, y_batch)

    if num_iterations % 10 == 0:
      # save predicted and ground truth images
      torchvision.transforms.ToPILImage(mode=None)(image_prediction[0].squeeze()).save(f'./test_predictions/p_{num_iterations}.jpg')
      torchvision.transforms.ToPILImage(mode=None)(y_batch[0].squeeze()).save(f'./test_ground_truth/g_{num_iterations}.jpg')

    elapsed_time = time.time() - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    print(f'Iteration: {num_iterations} | Time: {formatted_time} | IoU: {iou_score}')
    num_iterations += 1

    iou_values.append(iou_score)

# plotting IoU
plt.figure(figsize=(10, 5))
plt.plot(iou_values, label='IoU', linestyle='--')
plt.title('Testing IoU')
plt.xlabel('Iteration')
plt.ylabel('IoU')
plt.legend()
plt.savefig('./iou_plot.png')
plt.close()