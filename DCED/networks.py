import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pywt
import numpy as np
from torchvision.utils import save_image
DEVICE = 'cuda:0'

## Initialization Function ##

def init_weights(net, init_type = 'kaiming', gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('conv') != -1 or classname.find('Linear') != -1):
            ### The find() method returns -1 if the value is not found
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
                ### notice: weight is a parameter object but weight.data is a tensor
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func) ### it is called for m iterating over every submodule of (in this case) net as well as net itself, due to the method call net.apply(…).

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,dilation=1,bias=True),
                nn.ReLU(inplace=True))
    def forward(self,x):
        
        x = self.conv(x)
        return x


class Dilated_conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Dilated_conv_block,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=2,dilation=2,bias=True),
                nn.ReLU(inplace=True))
    def forward(self,x):

        x = self.conv(x)
        return x


class deconv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(deconv_block,self).__init__()
        self.up = nn.Sequential(
                nn.ConvTranspose2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,dilation=1,bias=True),
                nn.ReLU(inplace=True))
    def forward(self,x):
        
        x = self.up(x)
        return x

class EDBlock(nn.Module):
    def __init__(self,ch_in=32,ch_out=32):
        super(EDBlock,self).__init__()
        self.DiConv = Dilated_conv_block(ch_in,ch_out)
        self.Deconv = deconv_block(ch_in,ch_out)
        self.Conv = conv_block(ch_in,ch_out)
    def forward(self,x):
        E1= self.DiConv(x)
        E2 = self.DiConv(E1)
        E3 = self.DiConv(E2)

        D1 = self.Deconv(E3)
        A = E1 + D1
        C = self.Conv(A)
        return(C)


class DCED(nn.Module):
    def __init__(self, img_ch = 1, output_ch = 1):
        super(DCED, self).__init__()
        #self.L1 = conv_block(ch_in=img_ch,ch_out=64)
        self.L1 = nn.Sequential(nn.Conv2d(4,64,kernel_size=3,stride=1,padding=2,dilation=2,bias=True), nn.ReLU(inplace=True))

        self.L2 = nn.Sequential(nn.Conv2d(64,32,kernel_size=3,stride=1,padding=2,dilation=2,bias=True), nn.ReLU(inplace=True))

        self.EDB = EDBlock(ch_in=32,ch_out=32)
        self.SingleConv = nn.Conv2d(in_channels = 32*4,out_channels=1,kernel_size=3,stride=1,padding=1,bias=True)
        self.UpSample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners = True) 
    def forward(self,In):
        In = self.UpSample(In)
        in_img = In.detach().cpu().squeeze(0)
        in_img = in_img.squeeze(0)
        #print(in_img.shape)
        coeffs = pywt.dwtn(in_img,'haar')
        aa = coeffs['aa']
        ad = coeffs['ad']
        da = coeffs['da']
        dd = coeffs['dd']
        in_img = np.stack([aa,ad,da,dd], axis=0)
        in_img = torch.tensor(in_img,requires_grad=True).unsqueeze(0)
        in_img = in_img.float().to(DEVICE)
        
        x = self.L1(in_img)
        
        x = self.L2(x)
        
        EDB1 = self.EDB(x)
        EDB2 = self.EDB(EDB1)
        EDB3 = self.EDB(EDB2)
        cat1 = torch.cat((x,EDB1),dim=1)
        cat2 = torch.cat((cat1,EDB2),dim=1)
        cat3 = torch.cat((cat2,EDB3),dim=1)
        L = self.SingleConv(cat3)
        #L = L.detach().cpu().squeeze(0)
        #d = {'aa':L[3],'ad':L[2],'da':L[1],'dd':L[0]}
        #epoch = 0
        #e=0
        #save_image(torch.tensor(L[0]),'./featureMaps/img0.jpg')
        #save_image(torch.tensor(L[1]),'./featureMaps/img1.jpg')
        #save_image(torch.tensor(L[2]),'./featureMaps/img2.jpg')
        #save_image(torch.tensor(L[3]),'./featureMaps/img3.jpg')

        #sr = torch.tensor(pywt.idwtn(d, 'haar'),requires_grad=True).unsqueeze(0).to(DEVICE)
        sr = self.UpSample(L)
        A = In + sr
        return A






























