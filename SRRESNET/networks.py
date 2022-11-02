import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from deform_conv_2d import DeformConv2D
from torch.nn import init
import pywt
import numpy as np
from torchvision.utils import save_image
DEVICE = 'cuda:1'

def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
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
    net.apply(init_func)


class ConcatPool2d(nn.Module):
    def __init__(self, sz):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], dim=1)


class ConvLreluBlock(nn.Module):
    def __init__(self, inf, outf, kernel, stride, padding, slope=0.3):
        super().__init__()
        self.conv = nn.Conv2d(inf, outf, kernel, stride, padding)
        self.lrelu = nn.LeakyReLU(negative_slope=slope, inplace=True)

    def forward(self, x):
        return self.lrelu(self.conv(x))


class ConvBlock(nn.Module):
    def __init__(self, inf, outf, kernel, stride, padding, slope=0.3):
        super().__init__()
        self.conv = nn.Conv2d(inf, outf, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(outf)
        self.lrelu = nn.LeakyReLU(negative_slope=slope, inplace=True)

    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, nf, kernel, stride, padding, slope=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, kernel, stride, padding)
        self.bn1 = nn.BatchNorm2d(nf)
        self.lrelu = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, kernel, stride, padding)
        self.bn2 = nn.BatchNorm2d(nf)

    def forward(self, x):
        z = self.lrelu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        return x + z


class ShiftAndScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.scale * x + self.shift


class SwitchableResBlock(nn.Module):
    def __init__(self, nf, kernel, stride, padding, slope=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, kernel, stride, padding)
        self.bn1 = nn.BatchNorm2d(nf)
        self.lrelu = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, kernel, stride, padding)
        self.bn2 = nn.BatchNorm2d(nf)
        self.sas = ShiftAndScale()

    def forward(self, x):
        z = self.lrelu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        return self.sas(x) + z


class UpConv(nn.Module):
    def __init__(self, inf, outf, kernel, stride, padding, factor, slope=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(inf, outf, kernel, stride, padding),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Upsample(scale_factor=factor),
            nn.Conv2d(outf, outf, kernel, stride, padding),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
        )

    def forward(self, x):
        x = self.trunk(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning




class Generator(nn.Module):

    def __init__(self, kernel, inc, outc, ngf, n_blocks, tanh=True):
        super().__init__()
        self.conv1 = ConvBlock(inc, ngf, kernel, 1, 1)
        self.sigmoid = nn.Sigmoid()
        resblocks = []
        for _ in range(n_blocks):
            resblocks.append(ResBlock(ngf, kernel, 1, 1))

        self.trunk = nn.Sequential(*resblocks)
        self.conv2 = ConvBlock(ngf, ngf, kernel, 1, 1)

        #if scheme == 'isotropic':
        #    self.upsample = UpConv(ngf, ngf * 2, kernel, 1, 1, (2.0, 2.0, 2.0))
        #elif scheme == 'anisotropic':
        #    self.upsample = UpConv(ngf, ngf * 2, kernel, 1, 1, (2.0, 1.0, 1.0))
        #else:
        #    raise ValueError(f'Scheme {scheme} not understood. Must be `isotropic` or `anisotropic`')
        self.Up = nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None)
        if tanh:
            self.final_conv = nn.Sequential(
                ConvBlock(ngf, outc, kernel, 1, 1),
                nn.Tanh()
            )
        else:
            self.final_conv = nn.Sequential(
                ConvBlock(ngf, outc, kernel, 1, 1)
            )
        
    def forward(self, x):
        x = self.Up(x)
        img = x.detach().cpu().squeeze(0)
        img = img.squeeze(0)
        coeffs = pywt.dwtn(img,'haar')
        aa = coeffs['aa']
        ad = coeffs['ad']
        da = coeffs['da']
        dd = coeffs['dd']
        in_img = np.stack([aa,ad,da,dd], axis=0)
        in_img = torch.tensor(in_img,requires_grad=True).unsqueeze(0)
        in_img = in_img.float().to(DEVICE)
        z = self.conv1(in_img)
        y = self.conv2(self.trunk(z))
        y = y + z
        #y = self.upsample(y)
        y = self.final_conv(y)
        L = y.detach().cpu().squeeze(0)
        L = np.array(L)
        #print('hhhhhhhhhhhhhhhhh',type(L))
        sub1 = aa#+L[0]
        sub2 = ad+L[1]
        sub3 = da+L[2]
        sub4 = dd+L[3]
        d = {'aa':sub1,'ad':sub2,'da':sub3,'dd':sub4}
        save_image(torch.tensor(L[0]),'./featureMaps/img0.jpg')
        save_image(torch.tensor(L[1]),'./featureMaps/img1.jpg')
        save_image(torch.tensor(L[2]),'./featureMaps/img2.jpg')
        save_image(torch.tensor(L[3]),'./featureMaps/img3.jpg')
         
        sr = torch.tensor(pywt.idwtn(d, 'haar'),requires_grad=True).unsqueeze(0).to(DEVICE)
        #sr = self.sigmoid(sr)
        #print(sr.shape)
        return sr



