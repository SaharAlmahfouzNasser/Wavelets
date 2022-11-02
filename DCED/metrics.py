import torch 
import numpy as np
import kornia
def PSNR(y_true,y_pred):
     psnr = kornia.losses.psnr(y_true,y_pred,1)
     psnr = torch.mean(psnr)
     
     return psnr



def SSIM(y_true,y_pred):
    y_true = torch.unsqueeze(y_true,0)
    y_pred = torch.unsqueeze(y_pred,0)
    ssim = kornia.losses.ssim(y_true,y_pred,window_size=3)
    ssim = torch.mean(ssim)
    
    return ssim
