import os
import re
import torch
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision.io import read_image

def file_key(f_name):
    return [int(x) for x in re.findall('\d+',f_name)]

class PairedDataset(Dataset):
    def __init__(self, img_dataset, wm_dataset):
        self.data1 = img_dataset
        self.data2 = wm_dataset

        if len(img_dataset) != len(wm_dataset):
            print(
                "Lengths of dataset1 and dataset 2 are not equal"
            )
            exit()
        
    def __getitem__(self, index):
        return self.data2[index][0], self.data1[index][0], self.data2[index][1]
    
    def __len__(self):
        return len(self.data1)

class WmLoader(Dataset):
    def __init__(self,data_dir,split=None):
        data = torch.load(data_dir)
        self.wm_mask = data[0]
        _,reals,imags = data[1]
        reals = reals.to(torch.cfloat)
        reals.imag = imags

        if split is not None:
            self.data = reals[split]
        else:
            self.data = reals

    def __getitem__(self, index):
        return self.wm_mask,self.data[index]
    
    def __len__(self):
        return len(self.data)

class ImgLoader(Dataset):
    def __init__(self,data_dir,split=None):
        img_fps = os.listdir(data_dir)
        if split is not None:
            img_fps = [fp for fp in img_fps if int(re.findall('\d+',fp)[0]) in split]
        img_fps.sort(key=file_key)
        self.img_fps = img_fps
        self.data = [] 
        self.labels = []
        for img_fp in img_fps:
            img = read_image(os.path.join(data_dir,img_fp)).type(torch.float32)
            self.data.append(img)
        self.data = torch.stack(self.data)
        self.labels = torch.tensor([int(re.findall('\d+',i)[0]) % 2 for i in img_fps],dtype=torch.float32)

    def __getitem__(self, index):
        return self.data[index],self.labels[index],self.img_fps[index]
    
    def __len__(self):
        return len(self.data)
    
class LatentLoader(Dataset):
    def __init__(self,data_dir,split=None):
        latent_path = data_dir
        data = torch.load(latent_path)
        img_labels = data[0]
        latents = data[1]

        self.latent_max_val = latents.abs().max().detach().cpu().item()
        self.data = {}
        if split is not None:
            for i,latent in enumerate(latents):
                if img_labels[i] in split:
                    self.data[img_labels[i]] = latent[0]
        else:
            for i,latent in enumerate(latents):
                self.data[img_labels[i]] = latent[0]
        
        self.labels = torch.tensor([key % 2 for key in list(self.data.keys())],dtype=torch.float32)
        self.idxs = dict(enumerate(list(self.data.keys())))


    def __getitem__(self, index):
        return self.data[self.idxs[index]],self.labels[index],self.idxs[index]
    
    def __len__(self):
        return len(self.labels)
    
class ImgDataset(Dataset):
    def __init__(self,data_dirs,split=None):
        self.data = []
        self.labels = []
        cls = 0
        for data_dir in data_dirs:
            img_files = np.array(os.listdir(data_dir))
            img_files = img_files[split]
            for img_file in img_files:
                img = read_image(os.path.join(data_dir,img_file)).type(torch.float32)
                self.data.append(img)
            labels = torch.tensor([cls]*len(img_files))
            self.labels.append(labels)
            cls += 1
        self.data = torch.stack(self.data)
        self.labels = torch.concat(self.labels)

    def __getitem__(self, index):
        return self.data[index],self.labels[index]
    
    def __len__(self):
        return len(self.labels)
    
class LatentDataset(Dataset):
    def __init__(self, data_dirs: str, clss: list, split: list = None):
        self.data = []
        self.labels = []

        for dir,cls in zip(data_dirs, clss):
            idxs,latents = torch.load(dir)
            latents = latents[split] if split is not None else latents
            labels = torch.tensor([cls]*len(latents))
            self.data.append(latents)
            self.labels.append(labels)
            
        self.data = torch.concat(self.data)
        self.labels = torch.concat(self.labels)

    def __getitem__(self, index):
        return self.data[index],self.labels[index]

    def __len__(self):
        return len(self.labels)

class ImgDirDataset(Dataset):
    def __init__(self, data_dirs: str, clss: list, split: list = None):
        self.data = []
        self.labels = []

        for dir,cls in zip(data_dirs, clss):
            img_names = np.array(os.listdir(dir))
            img_names = img_names[split] if split is not None else img_names
            for img_name in img_names:
                img = read_image(os.path.join(dir, img_name)).type(torch.float32)
                self.data.append(img)
            labels = torch.tensor([cls]*len(img_names))
            self.labels.append(labels)
        self.data = torch.stack(self.data)
        self.labels = torch.concat(self.labels)

    def __getitem__(self, index):
        return self.data[index],self.labels[index]
    
    def __len__(self):
        return len(self.labels)
