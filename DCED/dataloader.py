import torch
from torch.utils.data import Dataset
import os 
import numpy as np
import nibabel as nib
from torchvision import transforms as T
import random
import torch.nn.functional as F
from scipy.ndimage import zoom
import pywt
def NORM(img):
    Max = np.max(img)
    Min = np.min(img)
    img = (img-Min)/(Max-Min+1e-10)
    return img
class MRI_Dataset(Dataset):
    def __init__(self, lr_path, hr_path, transforms = None,NORM = False):
        super().__init__()
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.lr_img = nib.load(lr_path).get_fdata().transpose(3, 2, 1, 0)
        self.hr_img = nib.load(hr_path).get_fdata().transpose(3, 2, 1, 0)
        #self.lr_list = sorted(os.listdir(lr_path))
        #self.hr_list = sorted(os.listdir(hr_path))
        self.transforms = transforms
        self.NORM = NORM
        #self.up = torch.nn.Upsample(scale_factor=2, mode='trilinear',align_corners = True)
    def __len__(self):
        return self.lr_img.shape[0]

    def __getitem__(self,i):
        lr_img =self.lr_img[i,10,:,:]#[i,10:20,15:30,15:30]
        hr_img = self.hr_img[i,20,:,:]#[i,20:40,30:60,30:60]
        #coeffs = pywt.dwtn(lr_img,'haar')
        #aa = torch.tensor(coeffs['aa'])
        #ad = torch.tensor(coeffs['ad'])
        #da = torch.tensor(coeffs['da'])
        #dd = torch.tensor(coeffs['dd'])
        #lr_img = torch.stack([aa,ad,da,dd], dim=0).float()
        #lr_img = lr_img[np.newaxis,:]
        lr_img = NORM(lr_img)
        hr_img = NORM(hr_img)
        lr_img = torch.from_numpy(lr_img[np.newaxis,:]).float()
        hr_img = torch.from_numpy(hr_img[np.newaxis,:]).float()
        #print(lr_img.shape)
        #print(hr_img.shape)

        """
        lr_img = torch.from_numpy(self.lr_img)
        hr_img = torch.from_numpy(self.hr_img)
        print(lr_img.shape)
        if self.NORM == True:
            Max = torch.max(lr_img)
            Min = torch.min(lr_img)
            lr_img = (lr_img-Min)/(Max-Min+1e-8)
            Max = torch.max(hr_img)
            Min = torch.min(hr_img)
            hr_img = (hr_img-Min)/(Max-Min+1e-8)
        


        #lr_img = torch.unsqueeze(lr_img, 0)
        #hr_img = torch.unsqueeze(hr_img, 0)
        print(lr_img.shape)
        
        lr_img = lr_img.type(torch.FloatTensor)
        hr_img = hr_img.type(torch.FloatTensor)

        if self.transforms is not None:
            lr_img = self.transforms(lr_img)
            hr_img = self.transforms(hr_img)
        """
        return lr_img,hr_img


