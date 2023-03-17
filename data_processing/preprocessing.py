import os
import shutil
import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#Build a final dataset from the two input sets
def build_final_dataset(originalSet, extraSet):
    train = "../FinalDataset/train"
    val = "../FinalDataset/val"
    test = "../FinalDataset/test"

    #make our datasets
    if not os.path.isdir("../FinalDataset"):
        os.mkdir("../FinalDataset")
    if not os.path.isdir(train):
        os.mkdir(train)
    if not os.path.isdir(val):
        os.mkdir(val)
    if not os.path.isdir(test):
        os.mkdir(test)
    
    i = 0
    for filename in os.listdir(originalSet):
        if (os.path.isdir(originalSet + "/" + filename)):
            
            addClass(originalSet, filename, 0)
    for filename in os.listdir(extraSet):
        if (os.path.isdir(extraSet + "/" + filename)):
            addClass(extraSet, filename, 1)



    



    

def addClass(originalSet, className, check):
    dataset = originalSet + "/" + className
    numImgs = len([name for name in os.listdir(dataset) if os.path.isfile(dataset+"/"+name)])
    trainCount = 0.8*numImgs
    valCount = 0.1*numImgs
    counter = 1
    for filename in os.listdir(dataset):
        if counter < trainCount:
            dest = "/Users/jasnoorguliani/Documents/courses/aps360/APS360/FinalDataset/train"
            if not os.path.isdir(dest + "/" + className):
                os.mkdir(dest + "/" + className)
        elif counter < trainCount + valCount:
            dest = "/Users/jasnoorguliani/Documents/courses/aps360/APS360/FinalDataset/val"
            if not os.path.isdir(dest + "/" + className):
                os.mkdir(dest + "/" + className)
        else:
            curr = "test"
            dest = "/Users/jasnoorguliani/Documents/courses/aps360/APS360/FinalDataset/test"
            if not os.path.isdir(dest + "/" + className):
                os.mkdir(dest + "/" + className)   
        origFilename = filename
        if (check):   
            filename = filename[:-4] + "_" + filename[-4:]  
        open(dest+"/"+className + "/" +filename, "w")
        shutil.copy(dataset + "/" + origFilename, dest+"/"+className + "/" +filename)
        counter+=1
    return numImgs
        
def getLoaders():
    transform = transforms.Compose([
        transforms.Resize((512, 384)),
        transforms.RandomRotation([0, 360]),
        transforms.GaussianBlur(5),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor()]
    )
    absPath = os.path.abspath("FinalDataset")
    training_path = absPath + "/train"
    validation_path = absPath + "/val"
    testing_path = absPath + "/test"
    train_set = torchvision.datasets.ImageFolder(root=training_path, transform=transform)
    validation_set = torchvision.datasets.ImageFolder(root=validation_path, transform=transform)
    testing_set = torchvision.datasets.ImageFolder(root=testing_path, transform=transform)
    

    train_loader = torch.utils.data.DataLoader(train_set,num_workers =0, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_set,num_workers =0, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testing_set,num_workers =0, batch_size=64, shuffle=True)

    return train_loader, val_loader, test_loader






#build_final_dataset("/Users/jasnoorguliani/Documents/courses/aps360/APS360/TrashNet_original","/Users/jasnoorguliani/Documents/courses/aps360/APS360/GarbageNet_12_classes")




