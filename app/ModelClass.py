import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torchvision

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

    
class DownstreamModel(nn.Module):
    def __init__(self, n_classes):
        super(DownstreamModel, self).__init__()
        
                
        self.upstream_model = Model(n_classes = 3755)
        # checkpoint = torch.load("../OCR/models/ocr_pretrained_model_checkpoint/best_model.pt")
        # self.upstream_model.load_state_dict(checkpoint["model_state_dict"])
        
        in_features = self.upstream_model.fc.in_features

        self.upstream_model.fc = torch.nn.Linear(in_features=in_features, out_features=n_classes, bias = True)
    
    
    def forward(self, x):
        logits,probas = self.upstream_model(x)
        return logits,probas


class CharClassificationModel(nn.Module):
    def __init__(self, n_classes):
        super(CharClassificationModel, self).__init__()

        self.upstream_model = Model(n_classes=3755)
        # checkpoint = torch.load(
        #     "../models/ocr_pretrained_model_checkpoint/best_model.pt")
        # self.upstream_model.load_state_dict(checkpoint["model_state_dict"])

        in_features = self.upstream_model.fc.in_features
        self.upstream_model.fc = torch.nn.Linear(
            in_features=in_features, out_features=n_classes, bias=True)

    def forward(self, x):
        logits, probas = self.upstream_model(x)
        return logits, probas


class GJImageModel(nn.Module):
    def __init__(self, n_classes):
        super(GJImageModel, self).__init__()

        self.base_model = torchvision.models.resnet34(pretrained=True)
        self.base_model.fc = torch.nn.Linear(
            in_features=512, out_features=n_classes, bias=True)

    def forward(self, x):
        logits = self.base_model(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
