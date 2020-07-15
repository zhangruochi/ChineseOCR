import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.utils as utils
from torchsummary import summary
import torchvision
from albumentations.core.transforms_interface import ImageOnlyTransform

from albumentations.pytorch import ToTensor
import albumentations as A
from tqdm import tqdm
import random
import numpy as np
import cv2
from collections import OrderedDict

from PIL import Image, ImageDraw, ImageFont

from matplotlib import pyplot as plt
import time
import os

import pickle as pkl
from pathlib import Path


# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

## Helpers
def show_batch(x,y,shape = None):
    """
    input: 
        x(Tensor[num_images, rows, columns]): images tensor
        y(array): labels
        shape(tuple): (rows,col) 
    output:
        grid of smaple images
    """

    if not shape:
        shape = (int(x.shape[0]**0.5), int(x.shape[0]**0.5))

    fig, axs = plt.subplots(nrows= shape[0], ncols=shape[1], figsize = (12,8))
    index = 0
    for row in axs:
        for ax in row:
            ax.imshow(x[index])
            ax.set_xlabel(y[index], )
            index+=1

    # plt.subplots_adjust(wspace = 0.2, hspace = 0.5) 
    fig.tight_layout()
    plt.show()


with open("chinese_labels","rb") as f:
    index2char = pkl.load(f)

font_root = Path("chinese_fonts")
font_paths = []
for _ in font_root.glob("*f"):
    font_paths.append(_)


background_img_paths = []

background_img_root = Path("background_img")
for _ in background_img_root.glob("*jpg"):
    background_img_paths.append(_)



## Hyper-parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

DEVICE = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
print("------train on {}-------".format(DEVICE))


## datasets
class CaptchaDataset(Dataset):
    
    def __init__(self, index2char, font_paths, train, transforms, multiple = 100):
        super(CaptchaDataset, self).__init__()
        
        self.index2char = index2char
        self.font_paths = font_paths
        self.transforms = transforms
        self.train = train
        
        self.labels = list(index2char.keys()) * multiple
        random.shuffle(self.labels)
        
    
    def __len__(self):
        return len(self.labels)
    

    def drawText(self, char, font_path):
        

        back_color = np.random.randint(0,255)

        img_arr = np.ones((48, 48, 3), dtype=np.int) * back_color
        img_draw = Image.fromarray(img_arr.astype('uint8'))
        draw = ImageDraw.Draw(img_draw)
        font = ImageFont.truetype(str(font_path), 22)
        
#         draw.text((13, 12), char,
#                       (255, 255, 255), font=font)
        draw.text((13, 12), char,
                  (np.random.randint(0,200),np.random.randint(0,200),np.random.randint(0,200)), font=font)
        image = cv2.cvtColor(np.asarray(img_draw), cv2.COLOR_RGB2BGR)
        
        return image
            
            
    def __getitem__(self, index):
        char = self.index2char[self.labels[index]]
        font_path = random.choice(self.font_paths)
        image = self.drawText(char, font_path) 
        
        if self.train:
            image = self.transforms(image = image)["image"]

        target = torch.tensor(self.labels[index])
        
        return image, target


## augmentations

class AddBackgroud(ImageOnlyTransform):
    """randomly add backgroud image in a sample.
    """

    def __init__(self, background_img_paths, p = 0.5):
        
        super(AddBackgroud, self).__init__()
        
        self.background_img_paths = background_img_paths
        self.back_img_path = str(random.choice(self.background_img_paths))
        self.p = p
        
    def  apply(self, image, **params):
        
        if np.random.rand() >= self.p:
            return image
        
        y1=random.randint(10,110)
        y2=y1+32
        x1=random.randint(10,261)
        x2=x1+32
        
        prev_bk = image[0][0].sum()
        image_bk = cv2.imread(self.back_img_path)[y1:y2, x1:x2]
        
        for y in range(32):
            for x in range(32):
                if image[y][x].sum() != prev_bk:
                    image_bk[y][x] = image[y][x]
        return image_bk


transforms = A.Compose([
                        A.Rotate(60,interpolation = cv2.INTER_LINEAR),
                        A.CenterCrop(32,32),
                        AddBackgroud(background_img_paths, p = 0.5),
                        A.HueSaturationValue(p=0.3),
                        A.RandomBrightnessContrast(p=0.3),
                        A.ToGray(p=0.5),
                        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                        ToTensor()
                    ])

dataset = CaptchaDataset(index2char, font_paths, train = True, transforms = transforms)


train_length = int(len(dataset) * 0.8)
val_length = len(dataset) - train_length

train_dataset, val_dataset = utils.data.random_split(dataset,[train_length, val_length])

train_dataloader = DataLoader(dataset = train_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             shuffle=True)

val_dataloader = DataLoader(dataset = val_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             shuffle=False)

dataset_loader = {"train": train_dataloader, "val": val_dataloader}


## Model

class Model(nn.Module):
    def __init__(self, n_classes):
        super(Model, self).__init__()
        
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        base_model = torchvision.models.resnet34(pretrained=False)
        checkpoint = torch.load("../models/pretrained_models/resnet34-333f7ec4.pth")
        base_model.load_state_dict(checkpoint)
#         for param in base_model.parameters():
#             param.requires_grad = False 

        base_model_layers = list(base_model.children()) 
        self.body = torch.nn.Sequential(*base_model_layers[4:9])
        
        
        self.fc = torch.nn.Linear(in_features=512, out_features=n_classes, bias = True)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = x.view(-1, x.shape[1])
        logits = self.fc(x)
        probas = F.softmax(logits, dim = 1)
        return logits,probas


model = Model(n_classes = len(index2char)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience = 1,
        threshold = 0.01,
        factor = 0.2,
        verbose = True,
        mode="max")


## Training

def compute_accuracy(model, data_loader, device):
    model.eval()
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def train_model(model, data_loader, optimizer, num_epochs,batch_size, device,metric_func, random_seed = 7):
    # Manual seed for deterministic data loader
    torch.manual_seed(random_seed)
    
    loss_list = []
    valid_acc_list = []
    best_valid_acc = 0
    
    for epoch in range(num_epochs):
        start = time.time()
        # set training mode
        model.train() 
        for batch_idx, (features, targets) in enumerate(data_loader["train"]):
            features = features.to(device)
            targets = targets.to(device)


            ## forward pass
            logits, probas = model(features)
            loss = F.cross_entropy(logits,targets)
            
            _, predicted_labels = torch.max(probas, 1)
            correct_pred = (predicted_labels == targets).sum()
            train_acc = correct_pred.float() / targets.size(0) * 100
            
            # backward pass
            # clear the gradients of all tensors being optimized
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### Login
            loss_list.append(loss.item())
            
            if batch_idx % 50 == 0:
                
                print ('Epoch: {0:03d}/{1:03d} | Batch {2:03d}/{3:03d} | Loss: {4:.3f} | Acc: {5:.3f}'.format(
                    epoch+1, num_epochs, batch_idx, 
                         len(train_dataset)//batch_size, loss, train_acc))
        
        end = time.time()
        with torch.set_grad_enabled(False):
            
            valid_acc = metric_func(model, data_loader["val"], device)
            
            print('Epoch: {0:03d}/{1:03d} | val acc: {2:.3f} % | time: {3:.3f} s'.format(
                  epoch+1, num_epochs, valid_acc, end-start))
            
            if not os.path.exists("../models/ocr_pretrained_model_checkpoint"):
                os.mkdir("../models/ocr_pretrained_model_checkpoint")
            
            if best_valid_acc <= valid_acc:
                best_valid_acc = valid_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'valid_acc': valid_acc,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, "../models/ocr_pretrained_model_checkpoint/best_model.pt")
            
    
            valid_acc_list.append(valid_acc)
        
        scheduler.step(valid_acc)
            
    checkpoint = torch.load("../models/ocr_model_checkpoint/best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
            
    return model, loss_list, valid_acc_list


model, loss_list, valid_acc_list = train_model(model, 
            dataset_loader, 
            optimizer, 
            NUM_EPOCHS, 
            device = DEVICE, 
            batch_size = BATCH_SIZE,
            metric_func = compute_accuracy)
