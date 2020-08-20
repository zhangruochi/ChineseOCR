import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.utils as utils
import torchvision
from torchsummary import summary

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

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

DEVICE = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

print("Train on {}".format(DEVICE))

root = Path("total_pic_patent")
char_set = set()
for direc in root.glob("*"):
    if direc.is_dir():
        for file in direc.glob("*.png"):
            if file.name != "origin.png":
                char = file.name[0]
                char_set.add(char)


class PatentOCRTestDataset(Dataset):

    def __init__(self, root, char_set, transforms=None):

        self.root = root
        self.transforms = transforms
        self.char_set = char_set

        self.char2index = {key: val for key,
                           val in zip(char_set, range(len(char_set)))}
        self.index2char = {val: key for key,
                           val in zip(char_set, range(len(char_set)))}

        self.samples = []
        self.labels = []

        for direc in root.glob("*"):
            if direc.is_dir():
                for file in direc.glob("*.png"):
                    if file.name != "origin.png":
                        self.samples.append(file)

        self.samples *= 5
        random.shuffle(self.samples)

        for p in self.samples:
            self.labels.append(self.char2index[p.name[0]])

    def __getitem__(self, index):

        img = self.samples[index]
        target = self.labels[index]
        img = cv2.imread(str(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]

        return img, torch.tensor(target)

    def __len__(self):
        return len(self.samples)


transforms = A.Compose([
    A.Resize(36, 36),
    A.Rotate(60, interpolation=cv2.INTER_LINEAR),
    A.CenterCrop(32, 32),
    A.HueSaturationValue(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.ToGray(p=0.3),
    A.OpticalDistortion(distort_limit=0.03, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensor()
])

dataset = PatentOCRTestDataset(root, char_set, transforms=transforms)


train_length = int(len(dataset) * 0.9)
val_length = len(dataset) - train_length

train_dataset, val_dataset = utils.data.random_split(
    dataset, [train_length, val_length])

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              #                              num_workers=4,
                              shuffle=True)

val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE,
                            #                              num_workers=4,
                            shuffle=False)

dataset_loader = {"train": train_dataloader, "val": val_dataloader}

if not os.path.exists("../models/patent_model_checkpoint"):
    os.makedirs("../models/patent_model_checkpoint")

with open("../models/patent_model_checkpoint/cache.pkl", "wb") as f:
    pkl.dump({"index2char": dataset.index2char}, f)


class Model(nn.Module):
    def __init__(self, n_classes):
        super(Model, self).__init__()

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=(1, 1), stride=(
                1, 1), padding=(0, 0), bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )

        base_model = torchvision.models.resnet34(pretrained=False)
        base_model_layers = list(base_model.children())

        self.body = torch.nn.Sequential(*base_model_layers[4:9])
        self.fc = torch.nn.Linear(
            in_features=512, out_features=n_classes, bias=True)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = x.view(-1, x.shape[1])
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


class PatentModel(nn.Module):
    def __init__(self, n_classes):
        super(PatentModel, self).__init__()

        self.upstream_model = Model(n_classes=3755)
        checkpoint = torch.load(
            "../models/ocr_pretrained_model_checkpoint/best_model.pt", map_location=DEVICE)
        self.upstream_model.load_state_dict(checkpoint["model_state_dict"])

        in_features = self.upstream_model.fc.in_features
        self.upstream_model.fc = torch.nn.Linear(
            in_features=in_features, out_features=n_classes, bias=True)

    def forward(self, x):
        logits, probas = self.upstream_model(x)
        return logits, probas


model = PatentModel(n_classes=len(char_set))
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=2,
    threshold=0.01,
    factor=0.2,
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


def train_model(model, data_loader, optimizer, num_epochs, batch_size, device, metric_func, random_seed=7):
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
            loss = F.cross_entropy(logits, targets)

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

            if batch_idx % 100 == 0:

                print('Epoch: {0:03d}/{1:03d} | Batch {2:03d}/{3:03d} | Loss: {4:.3f} | Acc: {5:.3f} % \n'.format(
                    epoch+1, num_epochs, batch_idx,
                    len(train_dataset)//batch_size, loss, train_acc))

                with open("train_log", "a") as f:
                    f.write('Epoch: {0:03d}/{1:03d} | Batch {2:03d}/{3:03d} | Loss: {4:.3f} | Acc: {5:.3f} \n'.format(
                        epoch+1, num_epochs, batch_idx,
                        len(train_dataset)//batch_size, loss, train_acc))

        end = time.time()
        with torch.set_grad_enabled(False):

            valid_acc = metric_func(model, data_loader["val"], device)

            print('Epoch: {0:03d}/{1:03d} | val acc: {2:.3f} % | time: {3:.3f} s'.format(
                  epoch+1, num_epochs, valid_acc, end-start))

            with open("train_log", "a") as f:
                f.write('Epoch: {0:03d}/{1:03d} | val acc: {2:.3f} % | time: {3:.3f} s \n'.format(
                    epoch+1, num_epochs, valid_acc, end-start))

            if not os.path.exists("../models/patent_model_checkpoint"):
                os.mkdir("../models/patent_model_checkpoint")

            if best_valid_acc < valid_acc:
                best_valid_acc = valid_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'valid_roc': valid_acc,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, "../models/patent_model_checkpoint/best_model.pt")

            valid_acc_list.append(valid_acc)

        scheduler.step(valid_acc)

    checkpoint = torch.load("../models/patent_model_checkpoint/best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, loss_list, valid_acc_list


model, loss_list, valid_acc_list = train_model(model,
                                               dataset_loader,
                                               optimizer,
                                               NUM_EPOCHS,
                                               device=DEVICE,
                                               batch_size=BATCH_SIZE,
                                               metric_func=compute_accuracy)
