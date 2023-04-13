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

import os
import matplotlib.pyplot as plt

import preprocessing

alexnet = torchvision.models.alexnet(pretrained = True)
use_cuda = True

base_data_path = "../FinalDataset"
base_alexnet_path = "../AlexNet/"

classes = os.listdir(base_data_path + "/train")
print(classes)

train_loader, val_loader, test_loader = preprocessing.getLoaders(batch_size=1)

def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "./model_checkpoints/model_{0}_bs{1}_lr{2}_epoch{3}".format(name, batch_size, learning_rate, epoch)
    return path


def save_features(data_loader, data_name):
    n = 0
    for img, label in data_loader:
        features = alexnet.features(img)
        features_tensor = torch.from_numpy(features.detach().numpy())

        path = f"{base_alexnet_path}{data_name}/{classes[label]}"
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(features_tensor.squeeze(0), f"{path}/feature_{n}.tensor")
        n += 1


class CNN_Transfer(nn.Module):
  # using 1 convolution layer, 1 pooling layer, and 2 fully connected layers.
  def __init__(self):
    self.name = "CNN_Transfer"
    super(CNN_Transfer, self).__init__()
    self.conv1 = nn.Conv2d(256, 256, 2) #in_channels, out_chanels, kernel_size
    self.pool = nn.MaxPool2d(2, 2) #kernel_size, stride =
    self.fc1 = nn.Linear(256*2*2, 32)
    self.fc2 = nn.Linear(32, len(classes))

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = x.view(-1, 256*2*2)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


def get_accuracy(model, train_loader, valid_loader, train=False):
    if train:
        data = train_loader
    else:
        data = valid_loader

    correct = 0
    total = 0
    for imgs, labels in data:
        
        #############################################
        #To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        #############################################

        output = model(imgs)
        
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total


def get_test_accuracy(model, test_loader):
    data = test_loader
    correct = 0
    total = 0
    for imgs, labels in data:
        
        
        #############################################
        # To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        #############################################

        
        output = model(imgs)
        
        # Select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total


def get_test_accuracy_for_label(model, test_loader, label):
    data = test_loader
    correct = 0
    total = 0
    for imgs, labels in data:
        
        
        #############################################
        # To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        #############################################

        if (labels[0] == label):
            output = model(imgs)
            
            # Select index with maximum prediction score
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += imgs.shape[0]
    return correct / total



def train_transfer(model, train_data, valid_data, batch_size=64, learning_rate=0.01, momentum=0.9, num_epochs=10):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    iters, losses, train_acc, val_acc = [], [], [], []

    # training
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):
          
          
            #############################################
            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()
            #############################################
            
              
            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

        # save the current training information
        iters.append(n)
        losses.append(float(loss)/batch_size)             # compute *average* loss
        train_acc.append(get_accuracy(model, train_loader, valid_loader, train=True)) # compute training accuracy 
        val_acc.append(get_accuracy(model, train_loader, valid_loader, train=False))  # compute validation accuracy
        n += 1

        # Checkpointing every epoch
        print(f"Epoch: {epoch}. Training acc: {train_acc[-1]}")
        path = get_model_name(model.name, batch_size, learning_rate, epoch)
        torch.save(model.state_dict(), path)

    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))


#save_features(train_loader, "train");
#save_features(val_loader, "val");
#save_features(test_loader, "test");

train_data_alexnet = torchvision.datasets.DatasetFolder(base_alexnet_path + "train", loader=torch.load, extensions=('.tensor'))
valid_data_alexnet = torchvision.datasets.DatasetFolder(base_alexnet_path + "val", loader=torch.load, extensions=('.tensor'))
test_data_alexnet = torchvision.datasets.DatasetFolder(base_alexnet_path + "test", loader=torch.load, extensions=('.tensor'))

model = CNN_Transfer()
if use_cuda and torch.cuda.is_available():
  model.cuda()
  print('CUDA is available!  Training on GPU ...')
else:
  print('CUDA is not available.  Training on CPU ...')
  
#proper model
train_transfer(model, train_data_alexnet, valid_data_alexnet, batch_size = 256, learning_rate = 0.01, num_epochs=25)


best_model = CNN_Transfer()
best_model_path = get_model_name(best_model.name, 256, 0.01, 24)
state = torch.load(best_model_path)
best_model.load_state_dict(state)
if use_cuda and torch.cuda.is_available():
    best_model.cuda()

test_dataloader = torch.utils.data.DataLoader(test_data_alexnet, batch_size=1, shuffle=True)
for i, label in enumerate(classes):
    test_accuracy = get_test_accuracy_for_label(best_model, test_dataloader, i)
    print(f"{label} has test acc: {test_accuracy}")

