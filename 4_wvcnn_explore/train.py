import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchmetrics
import numpy as np
import pywt
from sklearn.model_selection import StratifiedKFold
#from tqdm import tqdm

import pdb

def create_conv_block(din,dout,k,pool_stride):
    block = nn.Sequential(
        nn.Conv1d(din,dout,k),
        nn.BatchNorm1d(dout),
        nn.ReLU(),
        nn.MaxPool1d(pool_stride),
    )

    return block

class WFDBDataset(torch.utils.data.Dataset):
    def __init__(self,data,label):
        self.x = data
        self.y = label

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def train(args, device, data, label):
    torch.manual_seed(0)

    dwts = np.empty([data.shape[0],2,1127])
    for ii, data in enumerate(data):
        dwt_tmp = pywt.wavedec(data,'db2',level=4)
        dwts[ii,:,:] = np.concatenate((dwt_tmp[0], dwt_tmp[1]),0)

    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(dwts,label)

    for i, (train_index, test_index) in enumerate(skf.split(dwts, label)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        model = nn.Sequential(
            create_conv_block(2,10,5,3),
            create_conv_block(10,24,5,3),
            create_conv_block(24,50,5,3),
            create_conv_block(50,70,5,3),
            nn.Flatten(),
            nn.Linear(11*70,4)
        )
        if args.cuda:
            model.cuda()

        trainset = WFDBDataset(dwts[train_index].astype('float32'),label[train_index])
        testset = WFDBDataset(dwts[test_index].astype('float32'),label[test_index])
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.bs,shuffle=True)
        testloader = torch.utils.data.DataLoader(testset,batch_size=args.bs)

        optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120], gamma=0.1)
        loss = nn.CrossEntropyLoss()

        print("Start training")
        for epoch in range(1,args.epochs+1):
            print("Epoch " + str(epoch))
            do_epoch(args,model,device,trainloader,optimizer,loss,'train')
            do_epoch(args,model,device,testloader,optimizer,loss,'val')
            scheduler.step()

def do_epoch(args,model,device,loader,optimizer,loss_fct,traintest):
    total_loss = 0.0
    metric_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=4)
    metric_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=4, average=None)

    if traintest == 'train':
        model.train()
        total_loss = 0.0
        for batch_idx, (data,label) in enumerate(loader):
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = loss_fct(output,label)
            total_loss += loss

            acc = metric_acc(output, label)
            f1 = metric_f1(output, label)

            loss.backward()
            optimizer.step()
            
    else:
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data,label) in enumerate(loader):
                data, label = data.to(device), label.to(device)

                optimizer.zero_grad()

                output = model(data)
                loss = loss_fct(output,label)
                total_loss += loss

                acc = metric_acc(output, label)
                f1 = metric_f1(output, label)

    acc = metric_acc.compute()
    f1 = metric_f1.compute()
    f1_cinc17 = sum(f1[:3])/3

    print(f'{traintest}... Acc: {acc}, F1: {f1}, F1_CINC17: {f1_cinc17}, Loss: {total_loss}')

    metric_acc.reset()
    metric_f1.reset()
    