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

##### functions
# DWT, extract A4 and D4 DWT coefficients, a matrix with two rows is returned
def DWT_transfer_var(ds, lvl):
    ds_DWT = len(ds) * [[]]
    for i in range(len(ds_DWT)):
        tmp = pywt.wavedec(ds[i], 'db2', level=lvl)
        ds_DWT[i] = torch.cat(
            (torch.from_numpy(
                tmp[0]).unsqueeze(0), torch.from_numpy(
                tmp[1]).unsqueeze(0)), 0)
    return ds_DWT

# zero padding
def pad_sequence(sequences):
    max_len = 1125
    pad = torch.zeros([len(sequences), len(sequences[0]), max_len])
    for i, tensor in enumerate(sequences):
        length = tensor[0].size(0)
        if length > max_len:
            pad[i][..., :max_len] = tensor[:,:max_len]
        else:
            pad[i][..., :length] = tensor
    return pad

def crossValidation(k, ecg_sel, label_sel):
    ecg_fold = k * [[]]
    label_fold = k * [[]]
    num_Nor = 5076
    num_AF = 758
    num_Otr = 2415
    num_Nse = 279
    cnt = np.empty([k, 4])
    num = [math.ceil(num_Nor/k),math.ceil(num_AF/k),math.ceil(num_Otr/k),math.ceil(num_Nse/k)]

    for i in range(k):
        ecg_fold[i] = []
        label_fold[i] = np.empty(0)

    for j in range(k):
        cnt_Nor, cnt_AF, cnt_Otr, cnt_Nse = 4 * [0]
        no_Nor, no_AF, no_Otr, no_Nse = 4 * [0]

        for i in range(len(ecg_sel)):
            if label_sel[i] == 0:
                if num[0] * j <= cnt_Nor and cnt_Nor < num[0] * (j + 1):
                    ecg_fold[j] = ecg_fold[j] + [ecg_sel[i]]
                    label_fold[j] = np.concatenate((label_fold[j], [label_sel[i]]), axis=0)
                    no_Nor += 1
                cnt_Nor += 1

            elif label_sel[i] == 1:
                if num[1] * j <= cnt_AF and cnt_AF < num[1] * (j + 1):
                    ecg_fold[j] = ecg_fold[j] + [ecg_sel[i]]
                    label_fold[j] = np.concatenate((label_fold[j], [label_sel[i]]), axis=0)
                    no_AF += 1
                cnt_AF += 1

            elif label_sel[i] == 2:
                if num[2] * j <= cnt_Otr and cnt_Otr < num[2] * (j + 1):
                    ecg_fold[j] = ecg_fold[j] + [ecg_sel[i]]
                    label_fold[j] = np.concatenate((label_fold[j], [label_sel[i]]), axis=0)
                    no_Otr += 1
                cnt_Otr += 1

            elif label_sel[i] == 3:
                if num[3] * j <= cnt_Nse and cnt_Nse < num[3] * (j + 1):
                    ecg_fold[j] = ecg_fold[j] + [ecg_sel[i]]
                    label_fold[j] = np.concatenate((label_fold[j], [label_sel[i]]), axis=0)
                    no_Nse += 1
                cnt_Nse += 1

        cnt[j] = [no_Nor, no_AF, no_Otr, no_Nse]

    return ecg_fold, label_fold, cnt
    
# validation function
def testbench(model, loader, k, epoch, cri, best_F1, rep, total_epoch):
    correct = 0
    total = 0
    tpn,fpn,fnn,tpa,fpa,fna,tpo,fpo,fno = 9 * [0]

    model.eval()

    if loader == trainloader:
        typ = 'training'
    else:
        typ = 'validation'

    with torch.no_grad():
        for data in loader:
            inputs, targets = data[0].to(
                device, dtype=torch.float), data[1].to(
                device, dtype=torch.float)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            targets = targets.long()

            loss = cri(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # calc classes true positives, false positives and false negatives
            tpn += ((targets == 0) & (predicted == 0)).sum()
            fpn += ((targets != 0) & (predicted == 0)).sum()
            fnn += ((targets == 0) & (predicted != 0)).sum()

            tpa += ((targets == 1) & (predicted == 1)).sum()
            fpa += ((targets != 1) & (predicted == 1)).sum()
            fna += ((targets == 1) & (predicted != 1)).sum()

            tpo += ((targets == 2) & (predicted == 2)).sum()
            fpo += ((targets != 2) & (predicted == 2)).sum()
            fno += ((targets == 2) & (predicted != 2)).sum()

        # simplification from original F1 transformed to type I/II errors (see Wikipedia example)
        F1_normal = (2*tpn.float())/(2*tpn.float()+fnn.float()+fpn.float())
        F1_AF     = (2*tpa.float())/(2*tpa.float()+fna.float()+fpa.float())
        F1_other  = (2*tpo.float())/(2*tpo.float()+fno.float()+fpo.float())
        F1        = (F1_normal + F1_AF + F1_other) / 3

    print('==========================  %s  ==========================' % (typ))
    print('Total              Normal         AF              Other      Acc')
    print('%.2f               %.2f        %.2f           %.2f    %.2f%%'
          % (F1, F1_normal, F1_AF, F1_other, 100 * correct / total))

    # return average F1 of last 10 epochs
    if epoch > total_epoch - 11:
        best_F1 = best_F1 + F1
    if epoch == total_epoch - 1:
        best_F1 = best_F1 / 10

    return best_F1, F1

# training function
def train(model, loader, fold, noex, epoch, optimizer, cri):

    running_loss = 0.0
    model.train()

    # batch loop
    for i, data in enumerate(loader, 0):
        # transform data to device
        inputs, targets = data[0].to(device,dtype=torch.float),data[1].to(device, dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()

        # format data
        inputs = inputs.unsqueeze(1)  # add one dimension
        targets = targets.long()

        # forward and loss
        outputs = model(inputs)
        loss = cri(outputs, targets)

        # compute gradients and apply weight update
        loss.backward()
        optimizer.step()

        # print statistics
        Iter = len(loader)
        running_loss += loss.item()
        if i == Iter - 1:    # print the loss for every epoch
            # if model.score_type == 0:
            #     mask = model.get_neuron_score_mean(threshold,inputs)
            # elif model.score_type == 1:
            #     mask = model.get_neuron_score_entropy(threshold,inputs)
            # else:
            #     print('ERROR!')

            print('[Fold: %d no.repetition: %d epoch: %d, %5d] loss: %.5f' %
                  (fold, noex, epoch + 1, i + 1, running_loss / Iter))
            running_loss = 0.0


##### parameters
num_rep = 10  # num of repititions for each experiment
k = 5  # num of folds
num_epochs = 150
batchSize = 128
learningRate = 0.005
weightDecay = 0.01  # L2 regularization

##### main code
# import and format physionet data
data_raw = HDF5Dataset('../data/physionet2017_4classes.hdf5', True)
ecg = len(data_raw) * [[]]
label = np.zeros(len(data_raw))
for i, content in enumerate((data_raw)):
    tmp, label[i] = content
    ecg[i] = tmp[0]

# dividing the whole dataset into k folds, the ratio between each class in
# every fold is constant
ecg_fold, label_fold, num_item_per_class = crossValidation(k, ecg, label)
loader_params_train = {'batch_size': batchSize, 'shuffle': True, 'num_workers': 0}
loader_params_test = {'batch_size': 64, 'shuffle': False, 'num_workers': 0}

# define device (GPU support)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# empty result vectors
last_10_train = np.zeros([k, num_rep])
last_10_test = np.zeros([k, num_rep])
last_10_test_quant = np.zeros([k, num_rep])
F1_train = np.zeros([k, num_rep, num_epochs])
F1_test = np.zeros([k, num_rep, num_epochs])

# fold loop for cross validation
csvout_train = []
csvout_test = []
for j in range(k):

    # get test data from folds
    ecg_test = DWT_transfer_var(ecg_fold[j], 4)
    label_test = label_fold[j]
    ecg_test = pad_sequence(ecg_test)

    # get training data from folds
    ecg = copy.deepcopy(ecg_fold)
    label = copy.deepcopy(label_fold)
    ecg_train = []
    label_train = np.empty(0)
    del ecg[j]
    del label[j]
    for i in range(k - 1):
        ecg_train = ecg_train + ecg[i]
        label_train = np.concatenate((label_train, label[i]), 0)

    # augment training data (amplitude inverse) 
    flip_ecg = []
    flip_label = np.empty(0)
    for i, raw_ecg in enumerate((ecg_train)):
        tmp = torch.zeros([1, len(raw_ecg)], dtype=torch.long)
        tmp = - raw_ecg.to(torch.double)
        flip_ecg = flip_ecg + [(tmp)]
        flip_label = np.concatenate((flip_label, [label_train[i]]), 0)
    ecg_train = ecg_train + flip_ecg
    label_train = np.concatenate((label_train, flip_label), axis=0)

    # DWT preprocessing
    ecg_train = DWT_transfer_var(ecg_train, 4)
    ecg_train = pad_sequence(ecg_train)

    # setup dataloading for CNN training
    traindata = TrainDataset(ecg_train,label_train)
    testdata = TestDataset(ecg_test,label_test)
    trainloader = torch.utils.data.DataLoader(traindata, **loader_params_train)
    testloader = torch.utils.data.DataLoader(testdata, **loader_params_test)

    # set seed and randomizer options for reproducibility
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    prev_test_F1 = 0;
    # iteration loop
    for i in range(num_rep):

        # construct new network
        CNN = WVCNN4()
        CNN.to(device)

        # network options
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            CNN.parameters(),
            lr=learningRate,
            weight_decay=weightDecay)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 60, 90, 120], gamma=0.1)

        # epoch loop
        for epoch in range(num_epochs):

            # train for one epoch
            train(CNN, trainloader, j + 1, i + 1, epoch, optimizer, criterion)

            # calculate F1 scores
            last_10_train[j][i], F1_train[j][i][epoch] = testbench(
                CNN, trainloader, k, epoch, criterion, last_10_train[j][i], i, num_epochs)
            csvout_train.append([j,i,epoch,F1_train[j][i][epoch]])

            last_10_test[j][i], F1_test[j][i][epoch] = testbench(
                CNN, testloader, k, epoch, criterion, last_10_test[j][i], i, num_epochs)
            csvout_test.append([j,i,epoch,F1_test[j][i][epoch]])

            scheduler.step()

            # save model and state
            if prev_test_F1 < F1_test[j][i][epoch]:
                prev_test_F1 = F1_test[j][i][epoch]
                torch.save(CNN,"../results/snn_ref/model_fold" + str(j) + "_iter" + str(i) + ".pt")

with open('train_F1_scores.csv', 'w', newline='') as csvfile:
    wrtr = csv.writer(csvfile, delimiter=' ')
    wrtr.writerow(['No. Fold','No. Init Iteration','Epoch','F1 Score'])
    wrtr.writerows(csvout_train)

with open('test_F1_scores.csv', 'w', newline='') as csvfile:
    wrtr = csv.writer(csvfile, delimiter=' ')
    wrtr.writerow(['No. Fold','No. Init Iteration','Epoch','F1 Score'])
    wrtr.writerows(csvout_test)

print('Training: F1 ')
for p in range(len(last_10_train)):
    print('k=%.0f' % (p + 1))
    print(last_10_train[p])


print('Validation: F1 ')
for p in range(len(last_10_test)):
    print('k=%.0f' % (p + 1))
    print(last_10_test[p])



