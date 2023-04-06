import shutil
import rembg
from PIL import Image
import easygui as eg
import os
import torch
import random


'''
RCNN Dataset Class, loads segmented images with a bounding box
Labels are the following:
1: cardboard
2: glass
3: metal
4: paper
5: plastic
6: trash

get_item gets a single image and target pair
root is the path of the directory containing the img and bound folders
transforms is a list of transforms to be applied to the image


'''
class RCNNDataset(torch.utils.data.Dataset):
    def __init__(self,root, transforms=None):
        self.root = root
        self.transforms = transforms
        #load image files
        self.imgs = list(os.listdir(os.path.join(root, "images")))
        self.bounds = list(os.listdir(os.path.join(root, "bounds"))) #bounds are stored in the format (left, upper, right, lower)
    
    def __getitem__(self, idx):
        # load images and masks
        imgPath = os.path.join(root, "images", self.imgs[idx])
        boundTxt = self.imgs[idx][:-4] + ".txt"
        boundsPath = os.path.join(root, "bounds", boundTxt)
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
        labels = torch.as_tensor(labels, dtype=torch.float32)

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
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def singleDirectory():
    path = os.path.abspath("../FinalDataset")
    
    destPath = "../imgs"
    if not (os.path.isdir(destPath)):
        os.mkdir(destPath)

    train = path + "/train"
    val = path + "/val"
    test = path + "/test"
    for filename in os.listdir(train):
        if (os.path.isdir(train + "/" + filename)):
            addClass(train, destPath, filename)
    for filename in os.listdir(val):
        if (os.path.isdir(val + "/" + filename)):
            addClass(val, destPath, filename)
    for filename in os.listdir(test):
        if (os.path.isdir(test + "/" + filename)):
            addClass(test, destPath, filename)

def addClass(set, dest, filename):
    for name in os.listdir(set + "/" + filename):
        open(dest+"/"+name, "w")
        shutil.copy(set + "/" + filename + "/" + name, dest+"/"+name)
        

def getLabel(imagename):
    if 'cardboard' in imagename:
        return 1
    elif 'glass' in imagename:
        return 2
    elif 'metal' in imagename:
        return 3
    elif 'paper' in imagename:
        return 4
    elif 'plastic' in imagename:
        return 5
    elif 'trash' in imagename:
        return 6
    else:
        print("WTFFFFFFFFF")


#task: build composite images with a varying number of items, and store bounding boxes for those items
def buildRCNNImages():
    path = os.path.abspath("../imgs")
    output = "../RCNNSet"
    outputImgs = output + "/images"
    outputBounds = output + "/bounds"

    if not (os.path.isdir(output)):
        os.mkdir(output)
    if not (os.path.isdir(outputImgs)):
        os.mkdir(outputImgs)
    if not(os.path.isdir(outputBounds)):
        os.mkdir(outputBounds)
   
    counter = 0
    imageList = os.listdir(path)
    i = 0
    while i < len(imageList):
        flag = 0
        numObjs = random.randint(2, 5)
        print(i)

        boundFile = open(outputBounds + "/" + str(counter) + ".txt", "w")
        maxSize = (0,0)
        for k in range(0, numObjs):
            maxSize = max(Image.open(path + "/" + imageList[i+k]).size, maxSize)
        finalImg = Image.new('RGB', maxSize, "WHITE")
        for j in range(0, numObjs):
            currImg = Image.open(path + "/" + imageList[i+j])
            currOut = rembg.remove(currImg)
            bounds = currOut.getbbox()
            if (bounds != (0,0,512,384)):
                flag = 1
                boundFile.write(str(bounds[0]) + " " + str(bounds[1]) + " " + str(bounds[2]) + " "+ str(bounds[3]) + " ")
                
                label = getLabel(imageList[i+j])
                boundFile.write(str(label))
                boundFile.write("\n")


            finalImg.paste(currOut, (0,0), currOut)
        i = i + numObjs
        if (flag == 1):
            finalImg.convert('RGB').save(outputImgs + "/" + str(counter) + ".jpg", "JPEG")
        counter+=1






buildRCNNImages()
'''
input_path = "/Users/jasnoorguliani/Documents/courses/aps360/APS360/TrashNet_original/glass/glass1.jpg"
output_path = "/Users/jasnoorguliani/Documents/courses/aps360/APS360/TrashNet_original/glass/glass1_.png"
input = Image.open(input_path)
output = rembg.remove(input)
output.save(output_path)

input_path2 = "/Users/jasnoorguliani/Documents/courses/aps360/APS360/TrashNet_original/metal/metal1.jpg"
output_path2 = "/Users/jasnoorguliani/Documents/courses/aps360/APS360/TrashNet_original/metal/metal1_.png"
input2 = Image.open(input_path2)
output2 = rembg.remove(input2)
output2.paste(output, (0, 0), output)

new_image = Image.new("RGBA", output2.size, "WHITE") # Create a white rgba background
new_image.paste(output2, (0, 0), output2)              # Paste the image on the background. Go to the links given below for details.
new_image.convert('RGB').save(output_path2, "JPEG") 
'''