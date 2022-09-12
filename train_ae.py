import torch
import os
import pandas as pd
import scipy
import torchvision
from torch import optim,nn
from utils import StanfordCars
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Params

# # ae complexity
# img_size = 288
# transform = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((img_size,img_size)),
#     #T.RandomResizedCrop(image_size), # data augmentation
#     # T.RandomHorizontalFlip(),
#     torchvision.transforms.ToTensor()])
# train_batch = 128
# test_batch = 32
# lr=1e-2
# data_root = "/data/students/louis/standfordcars/standfordcars"
# epochs = 50

class AE(nn.Module):
    def __init__(self, **kwargs):
        super(AE, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=kwargs["input_shape"], out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        # self.enc5 = nn.Linear(in_features=32, out_features=16)
        # decoder 
        # self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=kwargs["input_shape"])
    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        # x = F.relu(self.nc5(x))
        # x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        return x
    
class CAE(nn.Module):
    
    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 img_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU,
                 ):
       
        super().__init__()
        c_hid = base_channel_size
        self.in_latent = int(img_size/np.power(2,3))

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
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
            x = self.encoder(x)
            x = self.linear(x)
            x = x.reshape(x.shape[0], -1, self.in_latent, self.in_latent)
            x = self.decoder(x)
            return x
        
        
        
def train_CAE(data_root, lr, transform =None, epochs=1, train_batch = 16):
    
    train_dataset = StanfordCars(root=data_root,split ="test",transform=transform)
    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=train_batch, shuffle=True, num_workers=4, pin_memory=True)

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu",0)
    print(device)

    # model = AE(input_shape=img_size*img_size*3).to(device)
    model = CAE(3,3,1000).to(device)

    # create an optimizer object
    optimizer = optim.Adam(model.parameters(), lr)

    # mean-squared error loss
    criterion = nn.MSELoss()


    restored_imgs = []
    losses = []
    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            
            # batch_features = batch_features.view(-1, img_size*img_size*3).to(device)
            batch_features = batch_features.to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(batch_features)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(train_loader)
        losses.append(loss)
        restored_imgs.append((epochs, batch_features, outputs))
    
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
    # save mode
    # store metrics