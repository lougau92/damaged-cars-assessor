# from tabnanny import verbose
# from this import d
from tabnanny import verbose
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
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

class Encoder(nn.Module):

    def __init__(self,
                 img_size,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(img_size*img_size, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):

    def __init__(self,
                 img_size,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*int(img_size*img_size/64)*c_hid),
            act_fn()
        )
        self.img_size = img_size

        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, int(self.img_size/8), int(self.img_size/8))
        x = self.net(x)
        return x


class Autoencoder(pl.LightningModule):

    def __init__(self,
                 height: int,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_channels: int = 3
                 ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(height, num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(height, num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, height, height)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=5,
                                                         min_lr=1e-4)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)


class GenerateCallback(pl.Callback):

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


 

# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html     
def train_CAE(
    dataset, # str: define which dataset to use
    data_root, 
    img_size, 
    lr,
    latent_dim,
    transforms, # data transformations
    epochs=1,
    train_batch = 16, # relative to the data loader
    num_workers =4,  # relative to the data loader
    shuffle = True,  # relative to the data loader
    config_path = "./no_path.yml",  # parameter file, the save model path will later be added on it
    checkpoint_path= "NONE",
    ):

    CHECKPOINT_PATH = "./models/CAE/"
    # print(transforms)
    if dataset =="standford":
        train_dataset =StanfordCars(root=data_root,split ="train",transform=transforms)
        v_dataset =StanfordCars(root=data_root,split ="validation",transform=transforms)
        test_dataset =StanfordCars(root=data_root,split ="test",transform=transforms)
    elif dataset=="cifar10":
        train_dataset =torchvision.datasets.CIFAR10(root="/data/students/louis/ciphar_10/", train=True,transform=transforms)
        test_dataset =torchvision.datasets.CIFAR10(root="/data/students/louis/ciphar_10/",train=False,transform=transforms)
        train_dataset, v_dataset = torch.utils.data.random_split(train_dataset, [45000, 5000])

    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=train_batch, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(v_dataset, batch_size=int(train_batch/2), shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(train_batch/2), shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu",0)
    print("Device used: ",device)

      # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}"),
                         accelerator='gpu' if str(device).startswith("cuda") else 0,
                         devices=1,
                         max_epochs=epochs,
                         enable_checkpointing=True,
                        #  enable_progress_bar =False,
                         auto_lr_find=False,
                         auto_scale_batch_size=True,
                         callbacks=[ModelCheckpoint(save_weights_only=True,
                         verbose=False,
                         filename="sample-{epoch:02d}-{val_loss:.2f}",
                         save_top_k=1,
                         monitor="val_loss"),
                                    GenerateCallback(get_train_images(8,train_dataset), every_n_epochs=1),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(height= img_size,base_channel_size=32, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
 # training saving

    model_name = strftime("CAE %d%b%Hh%M",gmtime(time.time()))

    dir = "./models/CAE/"+model_name+"/"
    os.mkdir(dir)

    PATH = dir+'/model.pth'
    torch.save(model.state_dict(), PATH) # saving model

    args = c.Config.get_args(config_path)
    args["model_paths"] = args["model_paths"] + [PATH]
    c.Config.store_args(c.Config, args = args,path = config_path) # saving model path
    shutil.copyfile(config_path,dir+'/params.yml')

def get_train_images(num,train_dataset):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)
