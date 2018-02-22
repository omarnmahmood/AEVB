#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:27:03 2018

@author: ac2123
"""
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

# Example of linear layer use
m = nn.Linear(20,30)
inp = Variable(torch.randn(128,20))
output = m(inp)
print(output.size())

dtype = torch.FloatTensor
# A bit of tutorial on tensors
N, D_in, H, D_out, = 64, 1000, 100, 10
# Create some tensor
x = torch.randn(D_in, H).type(dtype)
# See what happens if you make this a variable
xVar = Variable(x)
# See what the difference is if you use squeeze
xSqueeze = Variable(x.unsqueeze(0))


# Declare a tensor

z = torch.Tensor(2,1)
z[0,0] = 1
z[1,0] = 2
test = torch.mul(z,z.t())

# Declare another tesnor...

z = torch.Tensor(5,2)
z[:,0] = 1;
z[:,1] = 2;
