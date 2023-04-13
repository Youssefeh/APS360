import shutil
import rembg
from PIL import Image, ImageChops
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

#goal; shift image until it satisfies overlap concerns
def checkOverlap(bounds, bound, inputImg):
    num_transforms = 3 #only allow up to three transforms before quitting
    curr = 0
    success = False 

    while (success is False):
        if (curr == num_transforms):
            break
        areaNew = (bound[2] - bound[0])*(bound[3] - bound[1])
        changed = 0
        dirX = 0
        dirY = 0
        for existingBound in bounds:
            
            #calculate intersection of objects
            areaObj = (existingBound[2] - existingBound[0])*(existingBound[3] - existingBound[1])
            x_dist = (min(bound[2], existingBound[2]) -
              max(bound[0], existingBound[0]) )
  
            y_dist = (min(bound[3], existingBound[3]) -
              max(bound[1], existingBound[1]) )
            areaI = 0
            if x_dist > 0 and y_dist > 0:
                areaI = x_dist * y_dist
            IoU = areaI/(areaNew + areaObj - areaI)
            if (IoU > 0.8): #we have a bad overlap, bad
                return False
            elif(IoU > 0.5):
                
                #set offset direction
                if (changed == 0):
                    #print(bound[0], bound[1], bound[2], bound[3])
                    if (bound[2] + 20 > 512 and bound[0] -20 < 0):
                        dirX = 0
                    elif (bound[2] + 20 > 512):
                        dirX = -20
                    elif (bound[0] - 20 < 0):
                        dirX = 20
                    else:
                        if (bound[0] > existingBound[0]):
                            dirX = 20
                        else:
                            dirX = -20


                    if (bound[3] + 20 > 384 and bound[1] -20 < 0):
                        dirY = 0
                    elif (bound[3] + 20 > 384):
                        dirY = -20
                    elif (bound[1] - 20 < 0):
                        dirY = 20
                    else:
                        if (bound[1] > existingBound[1]):
                            dirY = 20
                        else:
                            dirY = -20                        
                    if (dirX == 0 and dirY == 0):
                        changed = 0
                        continue
                changed = 1
                inputImg = ImageChops.offset(inputImg, dirX, dirY)

                bound = inputImg.getbbox()
                curr+=1
                break
        if(changed == 0):
            break
    return True
            



    


#task: build composite images with a varying number of items, and store bounding boxes for those items
def buildRCNNImages():
    path = os.path.abspath("../imgs")
    output = "../RCNNSetUpdated"
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
    allBounds = []
    while i < len(imageList):
        numObjs = random.randint(2, 4)
        maxSize = (0,0)
        boundFile = open(outputBounds + "/" + str(counter) + ".txt", "w")
        for k in range(0, numObjs):
            maxSize = max(Image.open(path + "/" + imageList[i+k]).size, maxSize)
        baseImg = Image.open(path+"/" + imageList[i])
        resized = baseImg.resize(maxSize)
        bgRemoved = rembg.remove(resized)
        bounds = bgRemoved.getbbox()
        allBounds.append(bounds)
        boundFile.write(str(bounds[0]) + " " + str(bounds[1]) + " "+ str(bounds[2]) + " " + str(bounds[3]) + " ")
        label = getLabel(imageList[i])
        boundFile.write(str(label))
        boundFile.write("\n")

        for j in range(1, numObjs):
            currImg = Image.open(path + "/" + imageList[i+j])
            currOut = rembg.remove(currImg)
            bounds = currOut.getbbox()
            check = checkOverlap(allBounds, bounds, currOut)
            if (check is False):
                continue
            bounds = currOut.getbbox()
            allBounds.append(bounds)
            if (bounds != (0,0,512,384)):
                flag = 1
                boundFile.write(str(bounds[0]) + " " + str(bounds[1]) + " " + str(bounds[2]) + " "+ str(bounds[3]) + " ")
                
                label = getLabel(imageList[i+j])
                boundFile.write(str(label))
                boundFile.write("\n")


            resized.paste(currOut, (0,0), currOut)
            
        i = i + numObjs
        resized.convert('RGB').save(outputImgs + "/" + str(counter) + ".jpg", "JPEG")
        print("Saved " + str(counter) + " with " + str(numObjs) + " images: " + str(i))
        #if (counter == 10): break
        counter+=1






buildRCNNImages()
img = Image.open("/Users/jasnoorguliani/Downloads/IMG_6537.jpg")
size = img.size
output = rembg.remove(img)
new_image = Image.new("RGBA", size, "WHITE")
new_image.paste(output, (0,0), output)
new_image.convert('RGB').save("/Users/jasnoorguliani/Downloads/IMG_6537.jpg", "JPEG") 

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

'''
dataset = RCNNDataset()
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
'''
