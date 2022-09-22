import torch.nn.functional as F
from torch import optim,nn
from torchvision.models import *
import torch
import numpy as np
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar

class Encoder(nn.Module):
    def __init__(self, latent_dim,input_dim, acf=nn.GELU, base_filters = 3):
    
        super().__init__()

        self.encoder = nn.Sequential(
            ## 1st Convolutional Block
            nn.Conv2D(num_input_channels = 3, filters = base_filters , kernel_size = 5, strides = 2, padding = 1),
            acf(),
            
            ## 2nd Convolutional Block
            nn.Conv2d(filters = base_filters*2, kernel_size = 5, strides = 2, padding = 1),
            nn.BatchNorm2d(),
            acf(),            
            ## 3rd Convolutional Block
            nn.Conv2d(filters = base_filters* 4, kernel_size = 5, strides = 2, padding =1),
            nn.BatchNorm2d(),
            acf(),            
            
            ## 4th Convolutional Block
            nn.Conv2d(filters = base_filters* 8, kernel_size = 5, strides = 2, padding = 1),
            nn.BatchNorm2d(),
            acf(),            
     
            ## Flatten layer
            nn.Flatten(),
            
            ## 1st Fully Connected Layer
            nn.Linear(in_features = input_dim*input_dim*base_filters,out_features= 4096),
            nn.BatchNorm2d(),
            acf(),            
            
            ## 2nd Fully Connected Layer
            nn.Linear(in_features=4096, out_features=latent_dim)
            )
            
    def forward(self,x):
            x_hat = self.encoder(x)       
            return x_hat

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Generator(nn.Module):
    def __init__(self, latent_dim,input_dim, acf=nn.GELU, base_filters = 3):
    
        super().__init__()
        # self.reshape_layer = Reshape((8, 8, 256))
        num_classes = 2
      
        self.generator = nn.Sequential(

            # x = torch.concat((input_z_noise, input_label),1)
            
            nn.Linear(out_features = 2048, in_features = latent_dim + num_classes),
            acf(),            
            nn.Dropout(0.2),
            
            nn.Linear(in_features = 2048, out_features =256 * 8 * 8),
            nn.BatchNorm2d(256 * 8 * 8),
            acf(),            
            nn.Dropout(0.2),
            
            Reshape((8, 8, 256)),
        
            nn.Upsample(scale_factor = 2),
            nn.Conv2d( 8, 128, kernel_size = 5, padding = 1),
            nn.BatchNorm2d(128,momentum=0.8),
            acf(),            
            
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(128, 64, kernel_size = 5, padding = 1),
            nn.BatchNorm2d(64,momentum=0.8),
            acf(),            
            
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(64, 3, kernel_size = 5, padding = 1),
            nn.Sigmoid()
        )
    def forward(self,x):
                x_hat = self.generator(x)       
                return x_hat
            

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = nn
    def forward(self, x):
        return self.lambd(x)
         
def expand_label_input(x):
    x = K.expand_dims(x, axis = 1)
    x = K.expand_dims(x, axis = 1)
    x = K.tile(x, [1, 32, 32, 1])
    return x

class Discriminator(nn.Module):
    def __init__(self, latent_dim,input_dim, acf=nn.GELU, base_filters = 3):
    
        super().__init__()
    
    
        label_shape = (2, )
        # image_input = Input(shape = input_shape)
        # label_input = Input(shape = label_shape)
        
        self.layer1 =  nn.Sequential(

            nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1),
            acf()
        )

        self.discriminator =  nn.Sequential(      
            
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            acf(),            
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            acf(),            
            
            nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            acf(),            
            
            nn.Flatten(),
            nn.Linear(512*input_dim*input_dim,out_features = 1),
            nn.Sigmoid()
        )
        
    def forward(self,x):
                x = self.layer1(x)
                label_input1 = expand_label_input(torch.tensor(np.zeros(2,1)))
                x = nn.concat([x, label_input1], axis = 3)
                x_hat = self.discriminator(x)       
                return x_hat  

class GAN(LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = (channels, width, height)
        self.generator = Generator(latent_dim=self.hparams.latent_dim, input_dim=self.hparams.height)
        self.discriminator = Discriminator(latent_dim=self.hparams.latent_dim, input_dim=self.hparams.height)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)
    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

class CR_Model(nn.Module):
    def __init__(self, input_shape, acf=nn.GELU, base_filters = 3):
        
        super().__init__()
    
        self.resnet_model = inceptionResNetV2(include_top = False, weights = 'imagenet', 
                                        input_shape = input_shape, pooling = 'avg')
        
        
        # self.model = nn.Sequential(

        
        #     image_input = resnet_model.input
        #     x = resnet_model.layers[-1].output
        #     out = nn.Linear(128)
        #     embedder_model = Model(inputs = [image_input], outputs = [out])
            
        #     input_layer = Input(shape = input_shape)
            
        #     x = embedder_model(input_layer)
        #     Lambda(lambda x: K.l2_normalize(x, axis = -1))
        # )
        
            
    def forward(self,x):
            x = self.resnet_model(x)
            x = nn.Linear(128)(x)
            x = K.l2_normalize(x, axis = -1)
             
            return x_hat  
