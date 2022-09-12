import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision.io import read_image
import torchvision.transforms as transforms


class StanfordCars(Dataset):
    
    def __init__(self, root: str, split: str = "train", transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self._split = split
        self._base_folder = root

        if self._split == "train":
            self._images_base_path = self._base_folder + "/train/"
        else:
            self._images_base_path = self._base_folder + "/test/"

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
    
    
class CustomImageDataset(Dataset):
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
    
    dataset = CustomImageDataset(annotations_file=dataset_name,img_dirs=list_img_dirs,transform= dataset_transforms)   
    
    return dataset

def parse_transforms(trans_str_list,img_size):
    
    switch = {
        "pil" : transforms.ToPILImage(),
        "resize" : torchvision.transforms.Resize((img_size,img_size)),
        "randomflip" : transforms.RandomHorizontalFlip(),
        "randomcrop" : transforms.RandomResizedCrop(int(4*img_size/5)),
        "tensor" : torchvision.transforms.ToTensor()
        } 
    
    return torchvision.transforms.Compose([switch.get(x) for x in trans_str_list])
                   
