import torch
print(torch.__version__)
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
# from __future__ import annotations
import torch.nn as nn
import torch.optim as optim
from utils import build_dataset
import utils
from torchvision.models import *
import config as c

def train_model(
    data_path,
    lr,
    model_type,
    transforms, 
    epochs=1,
    train_batch = 16, 
    num_workers =4,
    shuffle = True,
    config_path = "./no_path.yml",
    factor=0.2,
    patience=5,
    min_lr=1e-3,
    mode="min"
    ):
    severity_train_path = data_path+"training/"
    severity_valid_path = data_path+"validation/"

    classes = None
    if classes == "severity":
        classes = {0:'01-minor',1:'02-moderate',2:'03-severe'}
    elif classes == "location":
        classes = {0:'front',1:'back',2:'side'}



    train_severity_ds = utils.build_dataset(severity_train_path,classes,"train_severity",transforms)
    valid_severity_ds = utils.build_dataset(severity_valid_path,classes,"valid_severity",transforms)


    trainloader = torch.utils.data.DataLoader(train_severity_ds, batch_size=train_batch, shuffle=shuffle,num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(valid_severity_ds, batch_size=train_batch/2, shuffle=shuffle,num_workers=num_workers)



    model_dict = {
    "dense169":densenet169(pretrained =True),
    "vgg16":vgg16(pretrained = True),
    "vgg19":vgg19(pretrained = True),
    "resnet50":resnet50(pretrained=True)

    # object detection
    # "fcnn":fasterrcnn_resnet50_fpn(),
    # yolov3-darknet53(),
    # fssd-darknet53(),
    }
    model = model_dict[model_type]


    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html see train on gpu
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0


    os.mkdir("./models/"+model_name+"/")
    c.Config.store_args(c.Config, args = metrics,path = "./models/"+model_name+"/metrics.yml")

    PATH = './models/'+model_name+'/model.pth'
    torch.save(model.state_dict(), PATH)

    args = c.Config.get_args(config_path)
    args["model_paths"] = args["model_paths"] + [PATH]
    c.Config.store_args(c.Config, args = args,path = config_path)

def validate(data,model,device):
    vloss = 0
    for x,_ in iter(data):
        x = x.to(device)
        x_hat = model(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        vloss += loss.item()

return vloss/len(data)

  