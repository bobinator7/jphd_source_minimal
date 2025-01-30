import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchmetrics
import numpy as np
import pywt
from sklearn.model_selection import StratifiedKFold
import itertools
import copy
import os

from models import *

import pdb

import matplotlib.pyplot as plt

class WFDBDataset(torch.utils.data.Dataset):
    def __init__(self,data,label):
        self.x = data
        self.y = label

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
def get_pat_split(dwts,label,patidx,tol=0.04):

    pats = np.unique(patidx)

    split = []

    for comb in itertools.combinations(pats,2):

        val = np.isin(patidx,comb)
        n_val_normal = (label[val]==0).sum()
        n_val_total = len(label[val])
        frac = n_val_normal / n_val_total
        if (frac < (0.5 + tol)) and (frac > (0.5 - tol)):
            split.append((np.where(~val)[0],np.where(val)[0],comb))

    return split

def train(args, device, data, label, patidx):
    torch.manual_seed(0)

    dwts = np.empty([data.shape[0],2,315])
    for ii, dat in enumerate(data):
        dwt_tmp = pywt.wavedec(dat,'db2',level=4)
        dwts[ii,:,:] = np.concatenate((dwt_tmp[0], dwt_tmp[1]),0)

    spl = get_pat_split(dwts,label,patidx)   

    for ii, (train_index, test_index, pat_index) in enumerate(spl):
    # ii = 0
    # (train_index, test_index, pat_index) = spl[0]

        print(f"Comb {ii}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        model_ref = nn.Sequential(
            create_conv_block(2,10,5,3),
            create_conv_block(10,24,5,3),
            create_conv_block(24,50,5,3),
            create_conv_block(50,70,5,3),
            nn.Flatten(),
            nn.Linear(1*70,2)
        )

        if args.cuda:
            model_ref.cuda()

        trainset = WFDBDataset(dwts[train_index].astype('float32'),label[train_index])
        testset = WFDBDataset(dwts[test_index].astype('float32'),label[test_index])
        all = WFDBDataset(dwts.astype('float32'),label)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.bs,shuffle=True)
        testloader = torch.utils.data.DataLoader(testset,batch_size=args.bs)
        loader = torch.utils.data.DataLoader(all,batch_size=args.bs,shuffle=True)

        optimizer = optim.Adam(model_ref.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120], gamma=0.1)
        loss = nn.CrossEntropyLoss()


        pt_filename = f'model_ref_tr{ii}.pt'
        if os.path.exists(pt_filename):
            model_ref.load_state_dict(torch.load(pt_filename))
        else:
            print("Start training ref")
            for epoch in range(1,args.epochs+1):
                print("Epoch " + str(epoch))
                do_epoch(args,model_ref,device,trainloader,optimizer,loss,'train')
                do_epoch(args,model_ref,device,testloader,optimizer,loss,'val')
                scheduler.step()

        for param in model_ref.parameters():
            param.requires_grad = False

        if ii == 0:
            torch.save(model_ref.state_dict(), f'model_ref_tr{ii}.pt')

        for jj in range(4):

            blocks = []
            for kk, block_ref in enumerate(model_ref):
                block_frozen = copy.deepcopy(block_ref)

                blocks.append(block_frozen)
                if jj == kk:
                    blocks.append(CorrectionLayerInterChannel(block_frozen[-3].weight.shape[0]))

            model_cl = nn.Sequential(*blocks)

            #pdb.set_trace()
            optimizer = optim.Adam(model_cl[jj+1].parameters(),lr=args.lr*0.01,weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120], gamma=0.1)
            loss = nn.CrossEntropyLoss()

            print("Start training cl")
            for epoch in range(1,args.epochs+1):
                print("Epoch " + str(epoch))
                do_epoch(args,model_cl,device,trainloader,optimizer,loss,'val')
                do_epoch(args,model_cl,device,testloader,optimizer,loss,'val')

                do_epoch(args,model_cl,device,loader,optimizer,loss,'train')
                #do_epoch(args,model_ref,device,testloader,optimizer,loss,'train')

                do_epoch(args,model_cl,device,trainloader,optimizer,loss,'val')
                do_epoch(args,model_cl,device,testloader,optimizer,loss,'val')
                scheduler.step()

            if jj == 1:
                torch.save(model_cl.state_dict(), f'model_cl_tr{ii}_pos{jj}.pt')

        
def eval(args, device, data, label, patidx, cl_pos):
    model_ref = nn.Sequential(
            create_conv_block(2,10,5,3),
            create_conv_block(10,24,5,3),
            create_conv_block(24,50,5,3),
            create_conv_block(50,70,5,3),
            nn.Flatten(),
            nn.Linear(1*70,2)
        )
    model_ref.load_state_dict(torch.load('model_ref_tr0.pt'))
    
    blocks = []
    for kk, block_ref in enumerate(model_ref):
        block_frozen = copy.deepcopy(block_ref)
        for param in block_frozen.parameters():
            param.requires_grad = False
        blocks.append(block_frozen)
        if kk == cl_pos:
            blocks.append(CorrectionLayerInterChannel(block_frozen[-3].weight.shape[0]))

    model_cl = nn.Sequential(*blocks)
    model_cl.load_state_dict(torch.load('model_cl_tr0_pos1.pt'))
    #pdb.set_trace()

    #plt.plot(data[0,0])
    #plt.savefig('figure.png')

    dwts = np.empty([data.shape[0],2,315])
    for ii, dat in enumerate(data):
        dwt_tmp = pywt.wavedec(dat,'db2',level=4)
        dwts[ii,:,:] = np.concatenate((dwt_tmp[0], dwt_tmp[1]),0)

    spl = get_pat_split(dwts,label,patidx)
    ii = 0
    (train_index, test_index, pat_index) = spl[0]

    trainset = WFDBDataset(dwts[train_index].astype('float32'),label[train_index])
    testset = WFDBDataset(dwts[test_index].astype('float32'),label[test_index])
    all = WFDBDataset(dwts.astype('float32'),label)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.bs,shuffle=True)
    testloader = torch.utils.data.DataLoader(testset,batch_size=args.bs)
    loader = torch.utils.data.DataLoader(all,batch_size=args.bs,shuffle=True)

    optimizer = optim.Adam(model_ref.parameters(),lr=args.lr*0.01,weight_decay=args.weight_decay)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120], gamma=0.1)
    loss = nn.CrossEntropyLoss()

    model_ref_merged = merge_bn_layers(model_ref)
    model_cl_merged = merge_bn_layers(model_cl)

    model_ref_quant = quantize_layers(model_ref_merged)
    model_cl_quant = quantize_layers(model_cl_merged)

    #pdb.set_trace()

    print('Model Reference')
    print('train set'); do_epoch(args,model_ref,device,trainloader,optimizer,loss,'val')
    print('test set'); do_epoch(args,model_ref,device,testloader,optimizer,loss,'val')

    print('Model CL')
    print('train set'); do_epoch(args,model_cl,device,trainloader,optimizer,loss,'val')
    print('test set'); do_epoch(args,model_cl,device,testloader,optimizer,loss,'val')

    print('Model Ref Merged')
    print('train set'); do_epoch(args,model_ref_merged,device,trainloader,optimizer,loss,'val')
    print('test set'); do_epoch(args,model_ref_merged,device,testloader,optimizer,loss,'val')

    print('Model CL Merged')
    print('train set'); do_epoch(args,model_cl_merged,device,trainloader,optimizer,loss,'val')
    print('test set'); do_epoch(args,model_cl_merged,device,testloader,optimizer,loss,'val')

    print('Model Ref Merg Quant')
    print('train set'); do_epoch(args,model_ref_quant,device,trainloader,optimizer,loss,'val')
    print('test set'); do_epoch(args,model_ref_quant,device,testloader,optimizer,loss,'val')

    print('Model CL Merg Quant')
    print('train set'); do_epoch(args,model_cl_quant,device,trainloader,optimizer,loss,'val')
    print('test set'); do_epoch(args,model_cl_quant,device,testloader,optimizer,loss,'val')

    export_parameters(model_ref_quant,'ref')
    export_parameters(model_cl_quant,'cl')

    dat_in = vec2int2c(data[0,0],12,8)
    with open('dat_in_ref.txt','w') as f:
        for it in dat_in:
            f.write(f'{it:03x}\n')

    pdb.set_trace()

def do_epoch(args,model,device,loader,optimizer,loss_fct,traintest):
    total_loss = 0.0
    metric_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)
    metric_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=2, average=None)

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

    print(f'{traintest}... Acc: {acc}, F1: {f1}, Loss: {total_loss}')

    metric_acc.reset()
    metric_f1.reset()
    
def export_parameters(model,filename,n_bits=12,n_frac=8):

    non_cl_w = []
    non_cl_b = []
    cl_param = []
    chn = []
    chn_accum = [0]
    w_offset = [0]
    b_offset = [0]
    n_cl = 0

    non_cl_w_kernel = []
    chnk = []
    chnk_accum = [0]
    wk_offset = [0]
    bk_offset = [0]
    
    for name, param in model.named_parameters():
        if 'weights' in name:
            cl_w_reshaped = torch.cat((param,torch.zeros([param.shape[0],1])),axis=1).reshape([24,-1,5])
            non_cl_w_kernel.append(cl_w_reshaped.permute([1,0,2]).flatten(end_dim=1).numpy())
            chnk.append(param.shape[1])
            chnk_accum.append(chnk_accum[-1]+param.shape[1])
            wk_offset.append(wk_offset[-1]+non_cl_w_kernel[-1].shape[0])
            bk_offset.append(bk_offset[-1])

            cl_param.append(param.permute([1,0]).flatten().numpy())
            n_cl = param.shape[0]
        elif 'weight' in name:
            if param.dim() == 3:
                
                non_cl_w_kernel.append(param.permute([1,0,2]).flatten(end_dim=1).numpy())
                chnk.append(param.shape[1])
                chnk_accum.append(chnk_accum[-1]+param.shape[1])
                wk_offset.append(wk_offset[-1]+non_cl_w_kernel[-1].shape[0])

                non_cl_w.append(param.permute([1,0,2]).flatten().numpy())
                chn.append(param.shape[1])
                chn_accum.append(chn_accum[-1]+param.shape[1])
                w_offset.append(w_offset[-1]+param.nelement())
                
            else:
                non_cl_w_kernel.append(param.T.reshape([2,-1,5]).flatten(end_dim=1).numpy())
                chnk.append(param.shape[1])
                chnk.append(param.shape[0])
                chnk_accum.append(chnk_accum[-1]+param.shape[1])
                chnk_accum.append(chnk_accum[-1]+param.shape[0])
                wk_offset.append(wk_offset[-1]+non_cl_w_kernel[-1].shape[0])

                non_cl_w.append(param.permute([1,0]).flatten().numpy())
                chn.append(param.shape[1])
                chn.append(param.shape[0])
                chn_accum.append(chn_accum[-1]+param.shape[1])
                chn_accum.append(chn_accum[-1]+param.shape[0])
                w_offset.append(w_offset[-1]+param.nelement())

        elif 'bias' in name:
            non_cl_b.append(param.numpy())
            b_offset.append(b_offset[-1]+param.nelement())

            bk_offset.append(bk_offset[-1]+param.nelement())

        print(name, param.shape)

    

    w_vec = np.concatenate(non_cl_w)
    wk_vec = np.concatenate(non_cl_w_kernel,axis=0)
    b_vec = np.concatenate(non_cl_b)

    #pdb.set_trace()

    w_int2c = vec2int2c(w_vec,n_bits,n_frac)
    wk_int2c = vec2int2c(wk_vec,n_bits,n_frac)
    b_int2c = vec2int2c(b_vec,n_bits,n_frac)

    if cl_param:
        cl_vec = np.concatenate(cl_param)
        cl_int2c = vec2int2c(cl_vec,n_bits,n_frac)

    for num_lay, lay in enumerate(model):
        if isinstance(lay,nn.Flatten):
            break

    chn_str = str(chn).replace('[','{').replace(']','}')
    chn_accum_str = str(chn_accum).replace('[','{').replace(']','}')
    w_offset_str = str(w_offset).replace('[','{').replace(']','}')
    b_offset_str = str(b_offset).replace('[','{').replace(']','}')

    #pdb.set_trace()

    with open('constants_'+filename+'.svh','w') as f:
        f.write(f'localparam CONV_N_LAY = {num_lay};\n')
        f.write(f'localparam FSM_STATE_WIDTH = CONV_N_LAY + 1;\n\n')
        f.write(f'localparam integer BLOCK_CHN [0:CONV_N_LAY+1] = {chn_str};\n')
        f.write(f'localparam integer CHN_ACCUM [0:CONV_N_LAY+2] = {chn_accum_str};\n')
        f.write(f'localparam CONV_K = {model[0][0].weight.shape[2]};\n')
        f.write(f'localparam FC_K = {int(model[-3][0].weight.shape[0]/model[-1][0].weight.shape[1])};\n')
        f.write(f'localparam COEFF_WIDTH = {n_bits};\n')
        f.write(f'localparam COEFF_FRA_WIDTH = {n_frac};\n')

        f.write(f'localparam integer WEIGHT_OFFSET [0:CONV_N_LAY+1] = {w_offset_str};\n')
        f.write(f'localparam integer BIAS_OFFSET [0:CONV_N_LAY+1] = {b_offset_str};\n')
        f.write(f'localparam integer N_WEIGHTS = WEIGHT_OFFSET[CONV_N_LAY+1];\n')
        f.write(f'localparam integer N_BIAS = BIAS_OFFSET[CONV_N_LAY+1];\n\n')

        f.write(f'localparam [N_WEIGHTS*COEFF_WIDTH-1:0] WEIGHTS = {{\n')
        for ii, it in enumerate(w_int2c):
            f.write(f"12'h{it:03x}")
            if ii < len(w_int2c)-1:
                f.write(',\n')
                    
        f.write(f'}};\n\n')

        f.write(f'localparam [N_BIAS*COEFF_WIDTH-1:0] BIAS = {{\n')
        for ii, it in enumerate(b_int2c):
            f.write(f"12'h{it:03x}")
            if ii < len(b_int2c)-1:
                f.write(',\n')

        f.write(f'}};\n\n')

        if cl_param:
            #pdb.set_trace()
            f.write(f'localparam integer N_CL = {n_cl};\n\n')
            f.write(f'localparam [N_CL*N_CL*COEFF_WIDTH-1:0] WCL = {{\n')
            for ii, it in enumerate(cl_int2c):
                f.write(f"12'h{it:03x}")
                if ii < len(cl_int2c)-1:
                    f.write(',\n')
            f.write(f'}};\n\n')


    # kernelwise memory for weights
    chnk_str = str(chnk).replace("[","'{32'd").replace("]","}").replace(", ",", 32'd")
    chnk_accum_str = str(chnk_accum).replace("[","'{32'd").replace("]","}").replace(", ",", 32'd")
    wk_offset_str = str(wk_offset).replace("[","'{32'd").replace("]","}").replace(", ",", 32'd")
    bk_offset_str = str(bk_offset).replace("[","'{32'd").replace("]","}").replace(", ",", 32'd")
    pdb.set_trace()

    with open('constantsk_'+filename+'.svh','w') as f:
        f.write(f'localparam FSM_STATE_WIDTH = {len(chnk)};\n\n')
        f.write(f'localparam integer BLOCK_CHN [0:FSM_STATE_WIDTH-1] = {chnk_str};\n')
        f.write(f'localparam integer CHN_ACCUM [0:FSM_STATE_WIDTH] = {chnk_accum_str};\n')
        f.write(f'localparam CONV_K = {model[0][0].weight.shape[2]};\n')
        f.write(f'localparam FC_K = {int(model[-3][0].weight.shape[0]/model[-1][0].weight.shape[1])};\n')
        f.write(f'//localparam COEFF_WIDTH = {n_bits};\n')
        f.write(f'localparam COEFF_FRA_WIDTH = {n_frac};\n')

        f.write(f'localparam integer WEIGHT_OFFSET [0:FSM_STATE_WIDTH] = {wk_offset_str};\n')
        f.write(f'localparam integer BIAS_OFFSET [0:FSM_STATE_WIDTH] = {bk_offset_str};\n')
        f.write(f'localparam integer N_WEIGHTS = WEIGHT_OFFSET[FSM_STATE_WIDTH];\n')
        f.write(f'localparam integer N_BIAS = BIAS_OFFSET[FSM_STATE_WIDTH];\n\n')


    with open('weight_'+filename+'.hex','w') as f:
        for it in wk_int2c:
            for it2 in it:
                f.write(f'{it2:03x}')
            f.write('\n')

    with open('bias_'+filename+'.hex','w') as f:
        for it in b_int2c:
            f.write(f'{it:04x}\n')

    #pdb.set_trace()
 

    return

def vec2int2c(vec,n_bits,n_frac):
    vec_int = (vec*2**n_frac).astype(int)
    vec_int[vec_int < 0] = vec_int[vec_int < 0] + 2**n_bits

    return vec_int