#!/usr/bin/env python
# coding: utf-8

##### libraries
## 3rd party
from __future__ import print_function
import pywt
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import pdb
import csv

from scipy.stats import entropy 

import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## custom
from utils.data_formatting import HDF5Dataset

##### classes
## network architecture
class WVCNN4(nn.Module):
    def __init__(self):
        super(WVCNN4, self).__init__()

        self.conv1 = nn.Conv2d(1, 5 * 2, (2, 5))
        self.conv2 = nn.Conv1d(5 * 2, 12 * 2, 5)
        self.conv3 = nn.Conv1d(12 * 2, 25 * 2, 5)
        self.conv4 = nn.Conv1d(25 * 2, 35 * 2, 5)
        self.pool1 = nn.MaxPool1d(3)
        self.pool2 = nn.MaxPool1d(3)
        self.fc = nn.Linear(11 * 35 * 2, 4)

        self.conv1_bn = nn.BatchNorm2d(5 * 2)
        self.conv2_bn = nn.BatchNorm1d(12 * 2)
        self.conv3_bn = nn.BatchNorm1d(25 * 2)
        self.conv4_bn = nn.BatchNorm1d(35 * 2)
        self.dropout = nn.Dropout(0.5)

    def _init_weight(self):
        init.xavier_uniform(self.conv1.weight)
        init.xavier_uniform(self.conv2.weight)
        init.xavier_uniform(self.conv3.weight)
        init.xavier_uniform(self.conv4.weight)
        init.xavier_uniform(self.fc.weight)

    def forward(self, x):
        # conv1 block
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)

        x = x.squeeze(2)
        x = self.pool1(x)

        # conv2 block
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)

        x = self.pool1(x)

        # conv3 block
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)

        x = self.pool1(x)
        x = self.dropout(x)

        # conv4 block
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = F.relu(x)

        x = self.pool2(x)
        x = self.dropout(x)

        # fc block
        x = x.view(-1, 11 * 35 * 2)
        x = self.fc(x)

        return x

## dataloader
class TrainDataset:
    def __init__(self,ecg_train,label_train):
        self.data = ecg_train
        self.label = label_train

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label

    def __len__(self):
        return len(self.data)

class TestDataset:
    def __init__(self,ecg_test,label_test):
        self.data = ecg_test
        self.label = label_test

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label

    def __len__(self):
        return len(self.data)

##### parameters
num_rep = 10  # num of repititions for each experiment
k = 5  # num of folds
num_epochs = 150
batchSize = 128
learningRate = 0.005
weightDecay = 0.01  # L2 regularization

##### main code
# import and format physionet data
data_raw = HDF5Dataset('physionet2017_4classes.hdf5', True)

# load network
model_path = 'snn_ref/model_fold4.pt'
CNN = torch.load(model_path)

# load sample data
no_sample = 2 #starting idx: 0
ecg_in = data_raw[no_sample][0][0]
if len(ecg_in >= 18000):
    data_in = ecg_in[0:18000]
else:
    data_in = np.zeros(18000)
    data_in[0:len(ecg_in)] = ecg_in

dwt = pywt.wavedec(ecg_in, 'db2', level=4)
dwt_out = torch.tensor([[[dwt[0],dwt[1]]]])
pdb.set_trace()

pred = CNN(dwt_out)
