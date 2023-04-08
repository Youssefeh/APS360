import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models
from PIL import Image

import sys, os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class RCNNDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        #load image files
        self.imgs = list(os.listdir(os.path.join(root, "images")))
        self.bounds = list(os.listdir(os.path.join(root, "bounds"))) #bounds are stored in the format (left, upper, right, lower)
    
    def __getitem__(self, idx):
        transform = transforms.ToTensor()

        # load images and masks
        imgPath = os.path.join(self.root, "images", self.imgs[idx])
        boundTxt = self.imgs[idx][:-4] + ".txt"
        boundsPath = os.path.join(self.root, "bounds", boundTxt)
        img = Image.open(imgPath).convert("RGB")


        boundFile = open(boundsPath, "r")
        bounds = boundFile.readlines()

        # get bounding box coordinates for each mask
        num_objs = len(bounds)
        boxes = []
        labels = []
        for i in range(num_objs):
            line = bounds[i]
            line.strip()
            indices = line.split()
            xmin = int(indices[0])
            xmax = int(indices[2])
            ymin = int(indices[1])
            ymax = int(indices[3])
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(indices[4]))
        

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
            print(f"image shape: {img.shape}")

        return img, target

    def __len__(self):
        return len(self.imgs) 

def get_transform():
    t = []
    # converts the image, a PIL image, into a PyTorch Tensor
    t.append(transforms.ToTensor())
    return transforms.Compose(t)

# create test data loader
test_dataset = RCNNDataset("../RCNNSet/test", get_transform())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if (device == torch.device('cuda')):
  print("Training on GPU!!!")
else:
  print("Training on CPU")

# create model
best_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

num_classes = 7 # 6 classes + background

in_features = best_model.roi_heads.box_predictor.cls_score.in_features

best_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# load state
epoch = 9
best_model_path = f"rcnn_epoch_{epoch}"
state = torch.load(best_model_path, map_location=device)
best_model.load_state_dict(state)


# CUDA

best_model.to(device)
best_model.eval()

colors = ['red', 'green', 'blue', 'purple', 'orange', 'black']
names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# demo output
demo_imgs = list(os.listdir('./Demo images/white other'))
with torch.no_grad():
    for img_name in demo_imgs:
        img_path = os.path.join('./Demo images/white other', img_name)
        img = Image.open(img_path).convert("RGB")
        transform = get_transform()
        img = transform(img)

        output = best_model([img])
        output = output[0]
        print(output)

        fig, ax = plt.subplots(1, 2)

        ax1 = ax[0]
        ax2 = ax[1]

        ax1.imshow(img.permute(1, 2, 0))

        ax2.imshow(img.permute(1, 2, 0))
        for i in range(len(output['boxes'])):
            if (output['scores'][i] < 0.4):
                continue;

            xmin, ymin, xmax, ymax = output['boxes'][i]
            label = output['labels'][i]
            if (label == 7):
                continue;
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor=colors[label-1], facecolor='none')

            ax2.text(xmin, ymin, names[label-1], color=colors[label-1])

            ax2.add_patch(rect)


        plt.savefig(f'Demo images/Results/white other/{img_name}')
        plt.close()
        #plt.show()


max_imgs = 10
with torch.no_grad():
    i = 0
    for img, target in test_loader:
        if i == max_imgs:
            break;

        output = best_model(img)
        output = output[0]
        print(output)

        fig, ax = plt.subplots(1, 2)

        ax1 = ax[0]
        ax2 = ax[1]

        ax1.set_xlabel('Input image and segmentation')
        ax2.set_xlabel('RCNN output')

        # render input data
        img = img[0]
        boxes = target['boxes'][0]
        labels = target['labels'][0]

        ax1.imshow(img.permute(1, 2, 0))

        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            label = labels[i]
            if (label == 7):
                continue;
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor=colors[label-1], facecolor='none')

            ax1.text(xmin, ymin, names[label-1], color=colors[label-1])

            ax1.add_patch(rect)


        # render model output
        ax2.imshow(img.permute(1, 2, 0))
        for i in range(len(output['boxes'])):
            if (output['scores'][i] < 0.75):
                continue;

            xmin, ymin, xmax, ymax = output['boxes'][i]
            label = output['labels'][i]
            if (label == 7):
                continue;
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor=colors[label-1], facecolor='none')

            ax2.text(xmin, ymin, names[label-1], color=colors[label-1])

            ax2.add_patch(rect)

        plt.show()
        
        i += 1
