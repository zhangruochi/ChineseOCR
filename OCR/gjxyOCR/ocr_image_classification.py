import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.utils as utils
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensor
import torchvision
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict
import PIL
from PIL import Image, ImageDraw, ImageFont
import cv2

from torchsummary import summary

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot as plt
import time
import os


from pathlib import Path
root = Path("image")


## check
for _ in root.glob("*"):
    if _.is_dir():
        dir_name = _.name
        for p in _.glob("*"):
            if p.name.split("_")[0] != dir_name:
                print(p)

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


## Hyper-parameters

BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

DEVICE = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
print("------train on {}-------".format(DEVICE))


## Datasets
class ImageDataset(Dataset):
    
    def __init__(self, image_path, transforms = None):
        super(ImageDataset, self).__init__()
        
        self.image_path = image_path
        self.classes = []
        self.all_images = []
        self.labels = []
        
        self.transforms = transforms
        
            
        for _ in self.image_path.glob("*"):
            if _.is_dir():
                dir_name = _.name
                self.classes.append(dir_name)
                for p in _.glob("*"):
                    self.all_images.append(p)
        
        
        self.class2index = {b:a for a,b in enumerate(self.classes)}
        self.index2class = {a:b for a,b in enumerate(self.classes)}
        
        random.shuffle(self.all_images)
        
        for p in self.all_images:
            self.labels.append( self.class2index[p.name.split("_")[0]])
            

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        
        img = cv2.imread(str(self.all_images[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = self.transforms(image = img)["image"]
        target = torch.tensor(self.labels[index])
        
        return img, target

    
    
class TestDataset(Dataset):
    
    def __init__(self, image_path, transforms = None):
        super(TestDataset, self).__init__()
        
        self.image_path = image_path
        self.classes = []
        self.all_images = []
        self.labels = []
        
        self.transforms = transforms
        
            
        for _ in self.image_path.glob("*"):
            if _.is_dir():
                dir_name = _.name
                self.classes.append(dir_name)
                for p in _.glob("*"):
                    self.all_images.append(p)
        
        
        self.class2index = {b:a for a,b in enumerate(self.classes)}
        self.index2class = {a:b for a,b in enumerate(self.classes)}
        
        random.shuffle(self.all_images)
        
        for p in self.all_images:
            self.labels.append( self.class2index[p.name.split("_")[0]])
            
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        
        img = cv2.imread(str(self.all_images[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = self.transforms(image = img)["image"]
        target = torch.tensor(self.labels[index])
        
        return img, target



train_transforms =  A.Compose([
                        A.ShiftScaleRotate(interpolation = cv2.INTER_NEAREST),
                        A.OneOf([A.VerticalFlip(p=0.3),
                                 A.HorizontalFlip(p=0.3)
                                ]),
                        A.RandomBrightnessContrast(p=0.5),
                        A.HueSaturationValue(p=0.5),
                        A.Cutout(2, p=0.5, max_h_size = 20, max_w_size = 20),
                        A.ToGray(p=0.2),
                        A.Resize(height = 224, width = 224, p=1),
                        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                        ToTensor()
                ])



test_transform = A.Compose([
    A.Resize(224,224),
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensor()
])


dataset = ImageDataset(root, transforms = train_transforms)


train_length = int(len(dataset) * 0.9)
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

test_dataset = TestDataset(root, transforms = test_transform)
test_dataloader = DataLoader(dataset = test_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             shuffle=False)

dataset_loader = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}

with open("../models/gj_image_model_checkpoint/cache.pkl", "wb") as f:
    pkl.dump({"index2char": dataset.index2char},f)



## Model

class GJImageModel(nn.Module):
    def __init__(self, n_classes):
        super(GJImageModel, self).__init__()
        
        self.base_model = torchvision.models.resnet34(pretrained=True)
#         for param in base_model.parameters():
#             param.requires_grad = False 

#         self.base_model.fc = torch.nn.Sequential(
#             torch.nn.Linear(in_features=512, out_features=1000, bias = True),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Dropout(p = 0.2),
#             torch.nn.Linear(in_features=1000, out_features=n_classes,bias =True)
#         )

        self.base_model.fc = torch.nn.Linear(in_features=512, out_features=n_classes, bias = True)
    
    
    def forward(self, x):
        logits = self.base_model(x)
        probas = F.softmax(logits, dim = 1)
        return logits,probas



model = GJImageModel(n_classes =  len(dataset.classes))
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience = 2,
        threshold = 0.01,
        factor = 0.2,
        verbose = True,
        mode="max")


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
    train_acc_list, valid_acc_list = [], []
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

            # backward pass
            # clear the gradients of all tensors being optimized
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### Login
            loss_list.append(loss.item())
            
            if batch_idx % 50 == 0:
                # print ('Epoch: {0:03d}/{1:03d} | Batch {2:03d}/{3:03d} | Loss: {4:.3f}'.format(
                #     epoch+1, num_epochs, batch_idx, 
                #          len(train_dataset)//batch_size, loss))

                with open("train_log", "a") as f:
                    f.write('Epoch: {0:03d}/{1:03d} | Batch {2:03d}/{3:03d} | Loss: {4:.3f}'.format(
                    epoch+1, num_epochs, batch_idx, 
                         len(train_dataset)//batch_size, loss))
        
        end = time.time()
        with torch.set_grad_enabled(False):
            train_acc = metric_func(model, data_loader["train"], device)
            valid_acc = metric_func(model, data_loader["val"], device)
            test_acc = metric_func(model, data_loader["test"], device)
            
            
            # print('Epoch: {0:03d}/{1:03d} train acc: {2:.3f} % | val acc: {3:.3f} % |  test acc: {4:.3f} % | time: {5:.3f} s'.format(
            #       epoch+1, num_epochs, train_acc, valid_acc, test_acc, end-start))

            with open("train_log","a") as f:
                f.write('Epoch: {0:03d}/{1:03d} train acc: {2:.3f} % | val acc: {3:.3f} % |  test acc: {4:.3f} % | time: {5:.3f} s'.format(
                  epoch+1, num_epochs, train_acc, valid_acc, test_acc, end-start))
            
            if not os.path.exists("../models/gj_image_model_checkpoint/"):
                os.mkdir("../models/gj_image_model_checkpoint/")
            
            if best_valid_acc <= valid_acc:
                best_valid_acc = valid_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'valid_acc': valid_acc,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, "../models/gj_image_model_checkpoint/best_model.pt")
            
            
            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)
        
        scheduler.step(valid_acc)
            
    checkpoint = torch.load("../models/gj_image_model_checkpoint/best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
            
    return model, loss_list, train_acc_list, valid_acc_list



model, loss_list, train_acc_list, valid_acc_list = train_model(model, 
            dataset_loader, 
            optimizer, 
            NUM_EPOCHS, 
            device = DEVICE, 
            batch_size = BATCH_SIZE,
            metric_func = compute_accuracy)