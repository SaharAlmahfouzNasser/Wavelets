import torch
from torch import nn
import torch.nn.functional as F

class MSE(nn.Module):
    def __init__(self):
        super(MSE,self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, Input, Output):
        mse =  self.mse(Input,Output)
        return mse

