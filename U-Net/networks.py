import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pywt
import numpy as np
from torchvision.utils import save_image
DEVICE = 'cuda:2'

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
                nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,dilation=1,bias=True),
                nn.ReLU(inplace=True))
    def forward(self,x):
        
        x = self.conv(x)
        return x
        
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
                nn.Upsample(scale_factor=1),
                nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)) ###inplace=True means that it will modify the input directly, without allocating any additional output.
    def forward(self,x):
        x = self.up(x)
        return x
class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.conv(x)
        return x
        
class dilated_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(dilated_conv,self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size = 1, stride=1,padding=0,dilation=2,bias=True)

    def forward(self,x):
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Up,self).__init__()
        self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True)) ###inplace=True means that it will modify the input directly, without allocating any additional output.
    def forward(self,x):
        x = self.up(x)
        return x
class U_Net(nn.Module):
    def __init__(self,img_ch = 4, output_ch = 4):
        super(U_Net,self).__init__()

        #self.Maxpool =nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.DC1 = dilated_conv(ch_in = 64,ch_out=64)
        self.Conv2 = conv_block(ch_in = 64,ch_out=128)
        self.DC2 = dilated_conv(ch_in = 128,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.DC3 = dilated_conv(ch_in = 256,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.DC4 = dilated_conv(ch_in = 512,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,padding=0)
        self.Up = Up(ch_in = 4,ch_out=1)
        self.Up_n = Up(ch_in = 1,ch_out=1)
        self.single_conv= single_conv(ch_in=4, ch_out=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        ## Encoder ##
        in_img = x.detach().cpu().squeeze(0)
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
        
        x1 = self.Conv1(in_img)
        #print(x1.shape)
        x2 = self.DC1(x1)
        #print(x2.shape)
        x2 = self.Conv2(x2)

        x3 = self.DC2(x2)
        x3 = self.Conv3(x3)

        x4 = self.DC3(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.DC4(x4)
        x5 = self.Conv5(x5)
        
        ## Decoder ##

        d5 = self.Up5(x5)
        #print(d5.shape,x4.shape)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)

        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)

        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)

        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
       
         
        """
        L = d1.detach().cpu().squeeze(0)
        L = np.array(L)
        #print('hhhhhhhhhhhhhhhhh',type(L))
        sub1 = L[0]
        sub2 = L[1]
        sub3 = L[2]
        sub4 = L[3]
        d = {'aa':sub1,'ad':sub2,'da':sub3,'dd':sub4}
        #save_image(torch.tensor(L[0]),'./featureMaps/img0.jpg')
        #save_image(torch.tensor(L[1]),'./featureMaps/img1.jpg')
        #save_image(torch.tensor(L[2]),'./featureMaps/img2.jpg')
        #save_image(torch.tensor(L[3]),'./featureMaps/img3.jpg')
         
        sr = torch.tensor(pywt.idwtn(d, 'haar'),requires_grad=True).unsqueeze(0).to(DEVICE)
        """
        sr = d1
        #print(sr.shape)
        #sr = sr.unsqueeze(0)
        sr =self.sigmoid( self.Up_n(self.Up(sr)))
        return sr
        
















