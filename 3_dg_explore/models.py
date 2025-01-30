import torch
import torch.nn as nn

import copy

import pdb

def create_conv_block(din,dout,k,pool_stride):
    block = nn.Sequential(
        nn.Conv1d(din,dout,k),
        nn.BatchNorm1d(dout),
        nn.ReLU(),
        nn.MaxPool1d(pool_stride),
    )
    return block

# class CorrectionLayerChannelWise(nn.Module):
#     def __init__(self, size_in):
#         super().__init__()
#         self.size = size_in
#         #weights a vector
#         weights = torch.Tensor(self.size)
#         weights = weights[:,None]
#         self.weights = nn.Parameter(weights) 
#         nn.init.normal_(self.weights, mean = 0.0, std = 1.0)

#     def forward(self, x):

#         #0th dimension of x (number of samples in a batch)
#         rep = x.data.size()[0]
#         #repeat weight vector for every batch sample
#         repeated = torch.repeat_interleave(torch.unsqueeze(self.weights,0), rep, dim = 0)
#         #multiply each row with the corresponding weight 
#         #(i-th row of matrix will be multiplied by i-th weight of the vector)
#         w_times_x = x * repeated 

#         return w_times_x

def round_halfup(x):
    return torch.ceil(torch.floor(2*x)/2)

def quant(x, bb):
    res = round_halfup(x * 2**bb) / 2**bb
    return res

class QuantLayer(nn.Module):
    def __init__(self, n_bits, n_frac, sat=False):
        super().__init__()
        self.n_bits = n_bits
        self.n_frac = n_frac
        self.sat = sat

    def _round_halfup(self,x):
        return torch.ceil(torch.floor(2*x)/2)

    def forward(self, x):
        res = self._round_halfup(x * 2**self.n_frac) / 2**self.n_frac

        # if self.sat:
        #     max_val = 

        return res

class CorrectionLayerInterChannel(nn.Module):
    def __init__(self, size_in):
        super().__init__()

        self.size = size_in

        weights = torch.eye(self.size)

        #weights = torch.Tensor(self.size,self.size)
        #nn.init.xavier_uniform_(weights) 
        #weights = torch.add (weights,torch.eye(self.size)) # non-zero diagonal

        self.weights = nn.Parameter(weights) 


    def forward(self, x):
        #0th dimension of x (number of samples in a batch)
        rep = x.data.size()[0]
        #repeat weight matrix for every batch sample
        repeated = torch.repeat_interleave(torch.unsqueeze(self.weights,0), rep, dim = 0)
        #weight matrix is multiplied by batch sample matrix
        w_times_x= torch.matmul(repeated, x)
        return w_times_x
    
def merge_bn_layers(model):

    blocks = []
    for block_ref in model:
        if isinstance(block_ref, nn.Sequential):
            din = block_ref[0].weight.shape[1]
            dout = block_ref[0].weight.shape[0]
            k = block_ref[0].weight.shape[2]
            pool_stride = block_ref[3].kernel_size
            conv_block = nn.Sequential(
                nn.Conv1d(din,dout,k),
                nn.ReLU(),
                nn.MaxPool1d(pool_stride),
            )

            rep = torch.prod(torch.tensor(list(block_ref[0].weight.shape[1:])))
            conv_block[0].weight = nn.Parameter(
                block_ref[0].weight \
                * torch.reshape(block_ref[1].weight.repeat_interleave(rep),block_ref[0].weight.shape) \
                / torch.sqrt(torch.reshape(block_ref[1].running_var.repeat_interleave(rep),block_ref[0].weight.shape))
            ,requires_grad=False)
            conv_block[0].bias = nn.Parameter(
                ((block_ref[0].bias - block_ref[1].running_mean) * block_ref[1].weight) \
                / torch.sqrt(block_ref[1].running_var) \
                + block_ref[1].bias
            ,requires_grad=False)

            blocks.append(copy.deepcopy(conv_block))

        else:
            blocks.append(copy.deepcopy(block_ref))
    
    model_merged = nn.Sequential(*blocks)

    return model_merged

def quantize_layers(model,n_bits=12,n_frac=8):

    blocks = []
    for block_ref in model:
        if isinstance(block_ref, nn.Sequential):
            conv_block = nn.Sequential(
                copy.deepcopy(block_ref[0]),
                QuantLayer(n_bits,n_frac),
                copy.deepcopy(block_ref[1]),
                copy.deepcopy(block_ref[2])
            )
            conv_block[0].weight = nn.Parameter(quant(conv_block[0].weight, n_frac),requires_grad=False)
            conv_block[0].bias = nn.Parameter(quant(conv_block[0].bias, n_frac),requires_grad=False)
            
            blocks.append(copy.deepcopy(conv_block))
        elif isinstance(block_ref,nn.Linear):
            quant_block = nn.Sequential(
                copy.deepcopy(block_ref),
                QuantLayer(n_bits,n_frac)
            )
            quant_block[0].weight = nn.Parameter(quant(quant_block[0].weight, n_frac),requires_grad=False)
            quant_block[0].bias = nn.Parameter(quant(quant_block[0].bias, n_frac),requires_grad=False)
            blocks.append(copy.deepcopy(quant_block))
        elif isinstance(block_ref,CorrectionLayerInterChannel):
            quant_block = nn.Sequential(
                copy.deepcopy(block_ref),
                QuantLayer(n_bits,n_frac)
            )
            quant_block[0].weights = nn.Parameter(quant(quant_block[0].weights, n_frac),requires_grad=False)
            #quant_block[0].bias = nn.Parameter(quant(quant_block[0].bias, n_frac),requires_grad=False)
            blocks.append(copy.deepcopy(quant_block))
        else:
            blocks.append(copy.deepcopy(block_ref))
    
    model_quant = nn.Sequential(*blocks)

    return model_quant