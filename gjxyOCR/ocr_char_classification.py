import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict

from PIL import Image, ImageDraw, ImageFont

from matplotlib import pyplot as plt
import time
import os
from pathlib import Path

test_image_path = Path("characters_cuts")

chars = set()
for p in test_image_path.glob("*.png"):
    chars.add(p.name.split(".")[0])
chars = list(chars)

font_paths = ["ttf/Fangsong.ttf", "ttf/Kaiti.ttf"]



## Hyper-parameters

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

DEVICE = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
print("------train on {}-------".format(DEVICE))


## Datasets

class CaptchaDataset(Dataset):
    
    def __init__(self, chars, font_paths, train, transforms, multiple = 50):
        super(CaptchaDataset, self).__init__()
        
        self.chars = chars
        self.char2index = {char:index for char, index in zip(self.chars, range(len(self.chars)))}
        self.index2char = {index:char for char, index in zip(self.chars, range(len(self.chars)))}
        
        self.multiple = multiple
        
        self.all_words = self.chars * multiple
        random.shuffle(self.all_words)        
        
        self.bgColor = (255, 255, 255)
        self.fontColor = (0, 0, 0)
        self.image_size = (42,42)
        self.font_size = 28
        
        self.font_paths = font_paths
        self.train = train
        self.transforms = transforms
        

    def __len__(self):
        return len(self.all_words)
    
    def drawText(self, pos, txt,font,image, fill):
        draw = ImageDraw.Draw(image)
        draw.text(pos, txt, font=font, fill= fill)
        del draw
    
    def __getitem__(self, index):
        word = self.all_words[index]
        font_path = random.choice(self.font_paths)
        img =  Image.new('RGB', self.image_size, self.bgColor)
        font = ImageFont.truetype(font_path, self.font_size)
        
        self.drawText((5,3), word, font, img, self.fontColor)

        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.functional.to_tensor(img)
            
        target = torch.tensor(self.char2index[word])
        
        return img, target

class TestDataset(Dataset):
    def __init__(self,chars, test_image_root, transfroms = None):
        
        self.chars = chars
        self.char2index = {char:index for char, index in zip(self.chars, range(len(self.chars)))}
        self.index2char = {index:char for char, index in zip(self.chars, range(len(self.chars)))}
        
        self.test_images = list(test_image_root.glob("*.png"))
        self.transforms = transfroms
        
    def __len__(self):
        return len(self.test_images)
    
    def __getitem__(self, idx):
        image = Image.open(self.test_images[idx])
        name = self.test_images[idx].name.split(".")[0]
        
        if self.transforms:
            image = self.transforms(image)
        
        target = self.char2index[name]
        return image,target


train_transforms = T.Compose([
                        T.RandomRotation(30,fill = (255,255,255)),
                        T.CenterCrop(size = 32),  
                        T.ColorJitter(contrast = 0.4, saturation = 0.4),
                        T.RandomGrayscale(p=1.0),
                        T.ToTensor(),
                        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),      
                    ])

test_data_transform = T.Compose([
    T.Resize(32),
    T.RandomGrayscale(p=1.0),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = CaptchaDataset(chars, font_paths, train = True, transforms = train_transforms,multiple = 30)
val_dataset = CaptchaDataset(chars, font_paths, train = True, transforms = train_transforms,multiple = 5)
test_dataset = TestDataset(chars, test_image_path,test_data_transform) 


train_dataloader = DataLoader(dataset = train_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             shuffle=True)

val_dataloader = DataLoader(dataset = val_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             shuffle=False)

test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size = BATCH_SIZE, 
                                                  num_workers=4,
                                                  shuffle = False)

dataset_loader = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}

with open("../models/gj_model_checkpoint/cache.pkl", "wb") as f:
    pkl.dump({"index2char": train_dataset.index2char},f)


class Model(nn.Module):
    def __init__(self, n_classes):
        super(Model, self).__init__()
        
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        base_model = torchvision.models.resnet34(pretrained=True)

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
    
    
class CharClassificationModel(nn.Module):
    def __init__(self, n_classes):
        super(CharClassificationModel, self).__init__()


        self.upstream_model = Model(n_classes = 3755)
        checkpoint = torch.load("../models/ocr_pretrained_model_checkpoint/best_model.pt")
        self.upstream_model.load_state_dict(checkpoint["model_state_dict"])
        
        
        in_features = self.upstream_model.fc.in_features
        self.upstream_model.fc = torch.nn.Linear(in_features=in_features, out_features=n_classes, bias = True)
    
    
    def forward(self, x):
        logits,probas = self.upstream_model(x)
        return logits,probas




model = CharClassificationModel(n_classes = len(chars))
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=1,
        threshold=0.01,
        factor = 0.2,
        verbose=True,
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

                print ('Epoch: {0:03d}/{1:03d} | Batch {2:03d}/{3:03d} | Loss: {4:.3f}'.format(
                    epoch+1, num_epochs, batch_idx, 
                         len(train_dataset)//batch_size, loss))

                with open("train_log", "a") as f:
                    f.write('Epoch: {0:03d}/{1:03d} | Batch {2:03d}/{3:03d} | Loss: {4:.3f} \n'.format(
                    epoch+1, num_epochs, batch_idx, 
                         len(train_dataset)//batch_size, loss))
        
        end = time.time()
        with torch.set_grad_enabled(False):
            train_acc = metric_func(model, data_loader["train"], device)
            valid_acc = metric_func(model, data_loader["test"], device)
            
            print('Epoch: {0:03d}/{1:03d} train acc: {2:.3f} % | val acc: {3:.3f} % | time: {4:.3f} s'.format(
                  epoch+1, num_epochs, train_acc, valid_acc, end-start))

            with open("train_log", "a") as f:
                f.write('Epoch: {0:03d}/{1:03d} train acc: {2:.3f} % | val acc: {3:.3f} % | time: {4:.3f} s \n'.format(
                  epoch+1, num_epochs, train_acc, valid_acc, end-start))

            
            if not os.path.exists("../models/gj_model_checkpoint/"):
                os.mkdir("../models/gj_model_checkpoint/")
            
            if best_valid_acc <= valid_acc:
                best_valid_acc = valid_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'valid_acc': valid_acc,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, "../models/gj_model_checkpoint/best_model.pt")
            
            
            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)
        
        scheduler.step(valid_acc)
            
    checkpoint = torch.load("../models/gj_model_checkpoint/best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
            
    return model, loss_list, train_acc_list, valid_acc_list

model, loss_list, train_acc_list, valid_acc_list = train_model(model, 
            dataset_loader, 
            optimizer, 
            NUM_EPOCHS, 
            device = DEVICE, 
            batch_size = BATCH_SIZE,
            metric_func = compute_accuracy)
