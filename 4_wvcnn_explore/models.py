import torch
import torch.nn as nn

def create_conv_block(din,dout,k,pool_stride):
    block = nn.Sequential(
        nn.Conv1d(din,dout,k),
        nn.BatchNorm1d(),
        nn.ReLU(),
        nn.MaxPool1d(pool_stride),
    )

def train(args, device, data, label):
    torch.manual_seed(0)

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