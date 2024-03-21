# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:04:24 2023

@author: a_kafaei
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import hilbert
import os
import sklearn
from sklearn.model_selection import train_test_split
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(torch.cuda.get_device_name(0))
import scipy
import time
from scipy.stats import entropy
from sklearn import metrics
import copy
from torch.utils.data import Subset
import tqdm
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
#%%
import torch
from torch import nn
import blitz
from blitz.modules import BayesianLinear
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from blitz.utils import variational_estimator





@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_dim, 64)
        self.blinear12 = BayesianLinear(64, 200)
        self.dropout = torch.nn.Dropout(p = 0.075)
        self.blinear2 = BayesianLinear(200, output_dim)
        
    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = F.leaky_relu((x_))
        x_ = self.dropout(self.blinear12(x_))
        x_ = F.leaky_relu(x_)
        return self.blinear2(x_)
    
    
#%%
regressor = BayesianRegressor(8, 2).cuda() # network with 8 inputs and 2 outputs
regressor.load_state_dict(torch.load('BNN_sample4096_fixed_th.pth.tar'))
regressor.eval()
#%%
A = sio.loadmat(r'./Sample_Data_4096/1.mat') # Features calculated from 4096 samples 
HK_gt = np.concatenate([A['log_a'],A['k']])
"features are R0.88,S0.88,K0.88,R0.72,S0.72,K0.72,X,U"
Fe = torch.from_numpy(A['Fe']).float().cuda()
Fe = Fe.squeeze().unsqueeze(0) # reshape to 1*8
out_bnn = []
t1 = time.time()
for p in range(100): #repeat 100 times to obtain the mean value and uncertainty
    out_bnn.append(regressor(Fe))
out_bnn = torch.stack(out_bnn)
t2 = time.time()
#%% printing the results
print('elapsed time is:',t2-t1)
alpha = out_bnn.squeeze()[:,0].cpu().data.numpy()
k = out_bnn.squeeze()[:,1].cpu().data.numpy()
print('mean estimate of log(alpha) is : ',alpha.mean())
print('variance of estimate of log(alpha) is : ',alpha.var())
print('Ground truth value of log(alpha) is :',HK_gt[0])
print('mean estimate of k is : ',k.mean())
print('variance of estimate of k is : ',k.var())
print('Ground truth value of k is :',HK_gt[1])