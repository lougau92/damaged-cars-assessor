import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision.io import read_image
import torchvision.transforms as transforms
import random
import shutil
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

# converts an array of strings to pytorch transforms
def parse_transforms(trans_str_list,img_size):
    
    switch = {
        "pil" : transforms.ToPILImage(),
        "resize" : torchvision.transforms.Resize((img_size,img_size)),
        "randomflip" : transforms.RandomHorizontalFlip(),
        "randomcrop" : transforms.RandomResizedCrop(int(4*img_size/5)),
        "tensor" : torchvision.transforms.ToTensor()
        } 

    return torchvision.transforms.Compose([switch.get(x) for x in trans_str_list])
 

class StanfordCars(Dataset):
    
    def __init__(self, transform, root: str, split: str = "train", target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self._split = split
        self._base_folder = root

        self._images_base_path = self._base_folder + split +"/"
    
        annotation = pd.DataFrame({"Filename":os.listdir(self._images_base_path)})

        self._samples = [
            (
                str(self._images_base_path + annot),
            )
            for annot in annotation["Filename"]
        ]


    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        """Returns pil_image and class_id for given index"""
        image_path = self._samples[idx]
        target = 0
        
        pil_image = Image.open(image_path[0]).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (pil_image, target)
    
    
class CarsDataset(Dataset):
    def __init__(self, annotations_file, img_dirs, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dirs = img_dirs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        class_val = -1;     count = 0
        class_vals = self.img_labels['Label'].value_counts(ascending=True).values
        
        while  (idx > class_val):
            class_val+=class_vals[count]
            count+=1
        count -=1   
        img_path = os.path.join(self.img_dirs[count], self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class CarsDataModule(LightningDataModule):
    def __init__(
        self,
        transforms = parse_transforms(['pil','resize','tensor'],128),
        data_dir: str = "/home/p63744/projects/louis/cars_data/internet_dataset_cleaned/data1a/",
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers


        self.dims = (1, 28, 28)
        self.num_classes = 2

        self.classes_dict = 'damaged'
        if self.classes_dict == "severity":
            self.classes_dict = {0:'01-minor',1:'02-moderate',2:'03-severe'}
        elif self.classes_dict == "location":
            self.classes_dict = {0:'front',1:'back',2:'side'}
        elif self.classes_dict == "damaged":
            self.classes_dict = {0:'00-damage',1:'01-whole'}

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.ds_train = build_dataset(self.data_dir+"training/",self.classes_dict,"train_damage",transforms)
            self.ds_val = build_dataset(self.data_dir+"validation/",self.classes_dict,"valid_damage",transforms)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.ds_test = build_dataset(self.data_dir+"test/",self.classes_dict,"test_damage",self.transforms)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers)


# function build espcially from loading the cars_data dataset, that organised the images in their corresponding class folder
def build_dataset(data_path,classes,dataset_name,dataset_transforms):
    labels_df = pd.DataFrame()
    list_img_dirs =[]
    i =0
    list_files = np.array([]);   list_labels = np.array([])
    
    for class_name in list(classes.values()):
        class_path = data_path + class_name +"/"
        list_img_dirs.append(class_path)
        for dir in os.listdir(class_path):
            list_files = np.append(list_files,dir)
            list_labels= np.append(list_labels,int(i))
        i+=1
        labels_df = pd.DataFrame({"Filename":list_files,"Label":list_labels.astype(int)})
    labels_df.to_csv(dataset_name, index = False)
    
    dataset = CarsDataset(annotations_file=dataset_name,img_dirs=list_img_dirs,transform= dataset_transforms)   
    
    return dataset

# converts an array of strings to pytorch transforms
def parse_transforms(trans_str_list,img_size):
    
    switch = {
        "pil" : transforms.ToPILImage(),
        "resize" : torchvision.transforms.Resize((img_size,img_size)),
        "randomflip" : transforms.RandomHorizontalFlip(),
        "randomcrop" : transforms.RandomResizedCrop(int(4*img_size/5)),
        "tensor" : torchvision.transforms.ToTensor()
        } 

    return torchvision.transforms.Compose([switch.get(x) for x in trans_str_list])
 
# moving files from a directory to another
def move_files (
    source = "/data/students/louis/cars_data/data3a/test/",
    dest = "/data/students/louis/cars_data/data3a/training/01-minor/"
    ):
    list_imgs = os.listdir(source)

    # test_set = random.sample(list_imgs,30)

    for file in list_imgs:

        if file.endswith("JPEG") or file.endswith("jpeg"):
            # construct full file path
            destination = dest + file
            # move file
            shutil.move(source+ file, destination)