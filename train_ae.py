from tabnanny import verbose
from this import d
import config as c
import torch
import os
import pandas as pd
import scipy
import torchvision
from torch import optim,nn
from utils import StanfordCars, parse_transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from time import gmtime, strftime
import time


class CAE(nn.Module):
    
    def __init__(self,
                 num_input_channels : int, # number of channels contained in the image (3 of RGB or 1 for grayscale)
                 base_channel_size : int, # factor relative to the complexity of the convolution
                 img_size : int, # it is assumed that the images are squared
                 latent_dim : int, # size of the latent space
                 act_fn : object = nn.GELU # activations between the convolution layers
                 ):
       
        super().__init__()
        c_hid = base_channel_size
        self.in_latent = int(img_size/np.power(2,3)) # value created for matching the different layer shapes

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*3*self.in_latent*self.in_latent),
            act_fn()
        )
    
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 288x288 => 144x144
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 144x144 => 72x72
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 72x72 => 36x36
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(in_features= 2*self.in_latent*self.in_latent*c_hid, out_features= latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 36x36 => 72x72
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 72x72 => 144x144
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 144x144 => 288x288
            nn.Tanh() # The input images is scaled between -1 and 1! must be rescaled
        )

    def forward(self, x):
            x = self.encoder(x)
            x = self.linear(x)
            x = x.reshape(x.shape[0], -1, self.in_latent, self.in_latent)
            x = self.decoder(x)
            return x
        
        
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html     
def train_CAE(
    dataset, # str: define which dataset to use
    data_root, 
    lr,
    img_size, 
    latent_dim,
    transforms, # data transformations
    epochs=1,
    train_batch = 16, # relative to the data loader
    num_workers =4,  # relative to the data loader
    shuffle = True,  # relative to the data loader
    config_path = "./no_path.yml",  # parameter file, the save model path will later be added on it
    factor=0.2, # relative to the optimiser scheduler
    patience=5,  # relative to the optimiser scheduler
    min_lr=1e-3, # relative to the optimiser scheduler
    mode="min",  # relative to the optimiser scheduler
    criterion = F.l1_loss
    ):

    pretrained_latent_dims = [64,128,256,384]

    dataset_list = {
        "standford":
        (StanfordCars(root=data_root,split ="train",transform=transforms),StanfordCars(root=data_root,split ="validation",transform=transforms)),
        "cifar10":
        "availabel soon"
        # (torchvision.datasets.CIFAR10(root=data_root, train=True,transform=transforms,download=True),torchvision.datasets.CIFAR10(root=data_root,train=False,transform=transforms,download=True))
    }
    (train_dataset,v_dataset) = dataset_list[dataset]

    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=train_batch, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    v_loader = torch.utils.data.DataLoader(v_dataset, batch_size=int(train_batch/2), shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu",0)
    print("Device used: ",device)

    model = CAE(3,3,img_size,latent_dim).to(device)

    # load a pretraind model if possible
    if latent_dim in pretrained_latent_dims:
        print(f"pretraining on cifar10_{latent_dim}..")
        pretrained_model_path = os.path.join("./models/", f"cifar10_{latent_dim}.pth")
        model.load_state_dict(torch.load(pretrained_model_path))

    # create an optimizer object
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode=mode,
                                                         factor=factor,
                                                         patience=patience,
                                                         min_lr=min_lr,
                                                         )



    losses = []
    start_time = time.time()

    for epoch in tqdm(range(epochs)):
        loss = 0
        min = 1
        for x, _ in train_loader:
            
            # batch_features = batch_features.view(-1, img_size*img_size*3).to(device)
            x = x.to(device)
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            x_hat = model(x)

            min = torch.min(x_hat).item()
            # compute training reconstruction loss
            train_loss = criterion(x, x_hat)#, reduction="none")
            # train_loss = train_loss.sum(dim=[1,2,3]).mean(dim=[0])
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(train_loader)
        val_loss = validate(v_loader,model,device,criterion)
        scheduler.step(val_loss)

        losses.append((epoch,loss,val_loss))
 
 # training saving

    model_name = strftime("CAE %d%b%Hh%M",gmtime(time.time()))

    metrics = {"losses":losses,"running time":time.time()-start_time}

    os.mkdir("./models/"+model_name+"/")
    c.Config.store_args(c.Config, args = metrics,path = "./models/"+model_name+"/metrics.yml")

    PATH = './models/'+model_name+'/model.pth'
    torch.save(model.state_dict(), PATH) # saving metrics

    args = c.Config.get_args(config_path)
    args["model_paths"] = args["model_paths"] + [PATH]
    c.Config.store_args(c.Config, args = args,path = config_path) # saving model

def validate(data,model,device,criterion):
    vloss = 0
    for x,_ in iter(data):
        x = x.to(device)
        x_hat = model(x)
        loss = criterion(x, x_hat)
        vloss += loss.item()

    return vloss/len(data)

