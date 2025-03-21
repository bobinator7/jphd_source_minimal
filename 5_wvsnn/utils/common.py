import numpy as np
import math
import pywt
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.set_default_tensor_type(torch.DoubleTensor)

from utils.data_formatting import HDF5Dataset

from tqdm import tqdm

# round function (round half up)
def round_halfup(x):
    return torch.ceil(torch.floor(2*x)/2)

##### classes
# original network architecture
"""
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
        #pdb.set_trace()
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
"""

# network architecture with normalization features
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
        
        self.normed = False
        self.input_lmbda = 1
        self.input_dlta = 0
        self.c1_lmbda = 1
        self.c1_dlta = 0
        self.c2_lmbda = 1
        self.c2_dlta = 0
        self.c3_lmbda = 1
        self.c3_dlta = 0
        self.c4_lmbda = 1
        self.c4_dlta = 0
        self.fc_lmbda = 1
        self.fc_dlta = 0
        
        #self.qtl = 0.999
        #self.norm_max_fcn = self.qtl_max
        #self.norm_min_fcn = self.qtl_min
        # calculate scaling and offset
        #qtl = 0.999

        #dwt_up = np.quantile(dwt_train.view(-1).numpy(),qtl)
        #dwt_lo = np.quantile(dwt_train.view(-1).numpy(),1-qtl)
        #dwt_lmbda = 1/(dwt_up-dwt_lo)
        #dwt_dlta = -dwt_lo*dwt_lmbda
        #c1_up = np.quantile(activation['conv1'].view(-1).numpy(),qtl)
        #c1_lo = np.quantile(activation['conv1'].view(-1).numpy(),1-qtl)
        #c1_lmbda = 1/(c1_up-c1_lo)
        #c1_dlta = -c1_lo*c1_lmbda
        #c2_up = np.quantile(activation['conv2'].view(-1).numpy(),qtl)
        #c2_lo = np.quantile(activation['conv2'].view(-1).numpy(),1-qtl)
        #c2_lmbda = 1/(c2_up-c2_lo)
        #c2_dlta = -c2_lo*c2_lmbda
        #c3_up = np.quantile(activation['conv3'].view(-1).numpy(),qtl)
        #c3_lo = np.quantile(activation['conv3'].view(-1).numpy(),1-qtl)
        #c3_lmbda = 1/(c3_up-c3_lo)
        #c3_dlta = -c3_lo*c3_lmbda
        #c4_up = np.quantile(activation['conv4'].view(-1).numpy(),qtl)
        #c4_lo = np.quantile(activation['conv4'].view(-1).numpy(),1-qtl)
        #c4_lmbda = 1/(c4_up-c4_lo)
        #c4_dlta = -c4_lo*c4_lmbda
        #fc_up = np.quantile(activation['fc'].view(-1).numpy(),qtl)
        #fc_lo = np.quantile(activation['fc'].view(-1).numpy(),1-qtl)
        #fc_lmbda = 1/(fc_up-fc_lo)
        #fc_dlta = -fc_lo*fc_lmbda
        
        self.conv1_beforenorm_weight = torch.zeros(self.conv1.weight.shape)
        self.conv1_beforenorm_bias = torch.zeros(self.conv1.bias.shape)
        self.conv2_beforenorm_weight = torch.zeros(self.conv2.weight.shape)
        self.conv2_beforenorm_bias = torch.zeros(self.conv2.bias.shape)
        self.conv3_beforenorm_weight = torch.zeros(self.conv3.weight.shape)
        self.conv3_beforenorm_bias = torch.zeros(self.conv3.bias.shape)
        self.conv4_beforenorm_weight = torch.zeros(self.conv4.weight.shape)
        self.conv4_beforenorm_bias = torch.zeros(self.conv4.bias.shape)
        self.fc_beforenorm_weight = torch.zeros(self.fc.weight.shape)
        self.fc_beforenorm_bias = torch.zeros(self.fc.bias.shape)
        
        self.conv1_beforequant_weight = torch.zeros(self.conv1.weight.shape)
        self.conv1_beforequant_bias = torch.zeros(self.conv1.bias.shape)
        self.conv2_beforequant_weight = torch.zeros(self.conv2.weight.shape)
        self.conv2_beforequant_bias = torch.zeros(self.conv2.bias.shape)
        self.conv3_beforequant_weight = torch.zeros(self.conv3.weight.shape)
        self.conv3_beforequant_bias = torch.zeros(self.conv3.bias.shape)
        self.conv4_beforequant_weight = torch.zeros(self.conv4.weight.shape)
        self.conv4_beforequant_bias = torch.zeros(self.conv4.bias.shape)
        self.fc_beforequant_weight = torch.zeros(self.fc.weight.shape)
        self.fc_beforequant_bias = torch.zeros(self.fc.bias.shape)
        
        self.norm_max_fcn = torch.max
        self.norm_min_fcn = torch.min
        
        self.quantize = False
        self.saturate = False
        self.q_actb = 12

    @torch.no_grad()
    def _init_weight(self):
        init.xavier_uniform(self.conv1.weight)
        init.xavier_uniform(self.conv2.weight)
        init.xavier_uniform(self.conv3.weight)
        init.xavier_uniform(self.conv4.weight)
        init.xavier_uniform(self.fc.weight)
        
    @torch.no_grad()
    def _q(self,x,b,sat=False):
        if ((x < 0).sum() != 0):
            assert True
        
        result = round_halfup(x * 2**b) / 2**b
        
        if sat:
            #if (result[result > 1].numel() != 0):
                #pdb.set_trace()
            max_val = 1-2**-b
            result[result > 1] = max_val
        
        return result 
    
    @torch.no_grad()
    def forward(self, x):
        if self.quantize:
            x = self._q(x,self.q_actb,sat=self.saturate)
        
        # conv1 block
        x = self.conv1(x)
        if not self.normed:
            #pdb.set_trace()
            x = self.conv1_bn(x)
        x = F.relu(x)
        
        if self.quantize:
            x = self._q(x,self.q_actb,sat=self.saturate)

        x = x.squeeze(2)
        x = self.pool1(x)

        # conv2 block
        x = self.conv2(x)
        if not self.normed:
            x = self.conv2_bn(x)
        x = F.relu(x)
        
        if self.quantize:
            x = self._q(x,self.q_actb,sat=self.saturate)

        x = self.pool1(x)

        # conv3 block
        x = self.conv3(x)
        if not self.normed:
            x = self.conv3_bn(x)
        x = F.relu(x)
        
        if self.quantize:
            x = self._q(x,self.q_actb,sat=self.saturate)

        x = self.pool1(x)
        x = self.dropout(x)

        # conv4 block
        x = self.conv4(x)
        if not self.normed:
            x = self.conv4_bn(x)
        x = F.relu(x)
        
        if self.quantize:
            x = self._q(x,self.q_actb,sat=self.saturate)

        x = self.pool2(x)
        x = self.dropout(x)

        # fc block
        x = x.view(-1, 11 * 35 * 2)
        x = self.fc(x)
        
        if self.quantize:
            x = self._q(x,self.q_actb,sat=self.saturate)

        return x
    
   
    
    # snn conversion custom functions
    @torch.no_grad()
    def update_weights_bn_integration(self):
        #pdb.set_trace()
        self.conv1.weight = nn.Parameter(self.conv1.weight
        * torch.reshape(self.conv1_bn.weight.repeat_interleave(self.conv1.weight.shape[0]),self.conv1.weight.shape)
        / torch.sqrt(torch.reshape(self.conv1_bn.running_var.repeat_interleave(self.conv1.weight.shape[0]),self.conv1.weight.shape)),requires_grad=False)
        self.conv1.bias = nn.Parameter(
        ((self.conv1.bias-self.conv1_bn.running_mean)
        * self.conv1_bn.weight) 
        / torch.sqrt(self.conv1_bn.running_var) 
        + self.conv1_bn.bias,requires_grad=False)
    
        #self.conv1_bn.running_mean = torch.zeros(self.conv1_bn.running_mean.shape)
        #self.conv1_bn.running_var = torch.ones(self.conv1_bn.running_var.shape)
        self.conv1_bn.reset_running_stats()
        self.conv1_bn.weight = nn.Parameter(torch.ones(self.conv1_bn.weight.shape),requires_grad=False)
        self.conv1_bn.bias = nn.Parameter(torch.zeros(self.conv1_bn.bias.shape),requires_grad=False)

        self.conv2.weight = nn.Parameter(self.conv2.weight
            * torch.reshape(self.conv2_bn.weight.repeat_interleave(torch.prod(torch.tensor(list(self.conv2.weight.shape[1:])))),self.conv2.weight.shape)
            / torch.sqrt(torch.reshape(self.conv2_bn.running_var.repeat_interleave(torch.prod(torch.tensor(list(self.conv2.weight.shape[1:])))),self.conv2.weight.shape)),requires_grad=False)
        self.conv2.bias = nn.Parameter(
            ((self.conv2.bias-self.conv2_bn.running_mean)
            * self.conv2_bn.weight) 
            / torch.sqrt(self.conv2_bn.running_var) 
            + self.conv2_bn.bias,requires_grad=False)

        #self.conv2_bn.running_mean = torch.zeros(self.conv2_bn.running_mean.shape)
        #self.conv2_bn.running_var = torch.ones(self.conv2_bn.running_var.shape)
        self.conv2_bn.reset_running_stats()
        self.conv2_bn.weight = nn.Parameter(torch.ones(self.conv2_bn.weight.shape),requires_grad=False)
        self.conv2_bn.bias = nn.Parameter(torch.zeros(self.conv2_bn.bias.shape),requires_grad=False)

        self.conv3.weight = nn.Parameter(self.conv3.weight
            * torch.reshape(self.conv3_bn.weight.repeat_interleave(torch.prod(torch.tensor(list(self.conv3.weight.shape[1:])))),self.conv3.weight.shape)
            / torch.sqrt(torch.reshape(self.conv3_bn.running_var.repeat_interleave(torch.prod(torch.tensor(list(self.conv3.weight.shape[1:])))),self.conv3.weight.shape)))
        self.conv3.bias = nn.Parameter(
            ((self.conv3.bias-self.conv3_bn.running_mean)
            * self.conv3_bn.weight) 
            / torch.sqrt(self.conv3_bn.running_var) 
            + self.conv3_bn.bias)

        #self.conv3_bn.running_mean = torch.zeros(self.conv3_bn.running_mean.shape)
        #self.conv3_bn.running_var = torch.ones(self.conv3_bn.running_var.shape)
        self.conv3_bn.reset_running_stats()
        self.conv3_bn.weight = nn.Parameter(torch.ones(self.conv3_bn.weight.shape))
        self.conv3_bn.bias = nn.Parameter(torch.zeros(self.conv3_bn.bias.shape))

        self.conv4.weight = nn.Parameter(self.conv4.weight
            * torch.reshape(self.conv4_bn.weight.repeat_interleave(torch.prod(torch.tensor(list(self.conv4.weight.shape[1:])))),self.conv4.weight.shape)
            / torch.sqrt(torch.reshape(self.conv4_bn.running_var.repeat_interleave(torch.prod(torch.tensor(list(self.conv4.weight.shape[1:])))),self.conv4.weight.shape)),requires_grad=False)
        self.conv4.bias = nn.Parameter(
            ((self.conv4.bias-self.conv4_bn.running_mean)
            * self.conv4_bn.weight) 
            / torch.sqrt(self.conv4_bn.running_var) 
            + self.conv4_bn.bias,requires_grad=False)

        #self.conv4_bn.running_mean = torch.zeros(self.conv4_bn.running_mean.shape)
        #self.conv4_bn.running_var = torch.ones(self.conv4_bn.running_var.shape)
        self.conv4_bn.reset_running_stats()
        self.conv4_bn.weight = nn.Parameter(torch.ones(self.conv4_bn.weight.shape),requires_grad=False)
        self.conv4_bn.bias = nn.Parameter(torch.zeros(self.conv4_bn.bias.shape),requires_grad=False)
        
        return
    
    @torch.no_grad()
    def print_norm_values(self):
        print(f'input: lmbda={self.input_lmbda},dlta={self.input_dlta}')
        print(f'conv1: lmbda={self.c1_lmbda},dlta={self.c1_dlta}')
        print(f'conv2: lmbda={self.c2_lmbda},dlta={self.c2_dlta}')
        print(f'conv3: lmbda={self.c3_lmbda},dlta={self.c3_dlta}')
        print(f'conv4: lmbda={self.c4_lmbda},dlta={self.c4_dlta}')
        print(f'fc: lmbda={self.fc_lmbda},dlta={self.fc_dlta}')
        return
    
    @torch.no_grad()
    def reset_norm_values(self):
        self.input_lmbda = 1
        self.input_dlta = 0
        self.c1_lmbda = 1
        self.c1_dlta = 0
        self.c2_lmbda = 1
        self.c2_dlta = 0
        self.c3_lmbda = 1
        self.c3_dlta = 0
        self.c4_lmbda = 1
        self.c4_dlta = 0
        self.fc_lmbda = 1
        self.fc_dlta = 0
        return
    
    @torch.no_grad()
    def update_weights_with_quant(self,b):
        self.conv1_beforequant_weight = self.conv1.weight.detach()
        self.conv1_beforequant_bias = self.conv1.bias.detach()
        self.conv2_beforequant_weight = self.conv2.weight.detach()
        self.conv2_beforequant_bias = self.conv2.bias.detach()
        self.conv3_beforequant_weight = self.conv3.weight.detach()
        self.conv3_beforequant_bias = self.conv3.bias.detach()
        self.conv4_beforequant_weight = self.conv4.weight.detach()
        self.conv4_beforequant_bias = self.conv4.bias.detach()
        self.fc_beforequant_weight = self.fc.weight.detach()
        self.fc_beforequant_bias = self.fc.bias.detach()
        
        w_ = self._q(self.conv1_beforequant_weight.detach(),b)
        self.conv1.weight = nn.Parameter(w_,requires_grad=False)
        b_ = self._q(self.conv1_beforequant_bias.detach(),b)
        self.conv1.bias = nn.Parameter(b_,requires_grad=False)
        
        w_ = self._q(self.conv2_beforequant_weight.detach(),b)
        self.conv2.weight = nn.Parameter(w_,requires_grad=False)
        b_ = self._q(self.conv2_beforequant_bias.detach(),b)
        self.conv2.bias = nn.Parameter(b_,requires_grad=False)
        
        w_ = self._q(self.conv3_beforequant_weight.detach(),b)
        self.conv3.weight = nn.Parameter(w_,requires_grad=False)
        b_ = self._q(self.conv3_beforequant_bias.detach(),b)
        self.conv3.bias = nn.Parameter(b_,requires_grad=False)
        
        w_ = self._q(self.conv4_beforequant_weight.detach(),b)
        self.conv4.weight = nn.Parameter(w_,requires_grad=False)
        b_ = self._q(self.conv4_beforequant_bias.detach(),b)
        self.conv4.bias = nn.Parameter(b_,requires_grad=False)
        
        w_ = self._q(self.fc_beforequant_weight.detach(),b)
        self.fc.weight = nn.Parameter(w_,requires_grad=False)
        b_ = self._q(self.fc_beforequant_bias.detach(),b)
        self.fc.bias = nn.Parameter(b_,requires_grad=False)
        
        return

    @torch.no_grad()
    def update_weights_with_norm_values(self):
        
        if self.normed:
            print('[WVCNN4 Info]: Network is already normed!')
            return
        
        
        self.conv1_beforenorm_weight = self.conv1.weight.detach()
        self.conv1_beforenorm_bias = self.conv1.bias.detach()
        self.conv2_beforenorm_weight = self.conv2.weight.detach()
        self.conv2_beforenorm_bias = self.conv2.bias.detach()
        self.conv3_beforenorm_weight = self.conv3.weight.detach()
        self.conv3_beforenorm_bias = self.conv3.bias.detach()
        self.conv4_beforenorm_weight = self.conv4.weight.detach()
        self.conv4_beforenorm_bias = self.conv4.bias.detach()
        self.fc_beforenorm_weight = self.fc.weight.detach()
        self.fc_beforenorm_bias = self.fc.bias.detach()
        
        
        w_ = self.conv1_beforenorm_weight.detach() * (self.c1_lmbda.detach()/self.input_lmbda.detach())
        self.conv1.weight = nn.Parameter(w_,requires_grad=False)
        b_ = (self.conv1_beforenorm_bias.detach() - torch.sum(self.conv1_beforenorm_weight.detach() * (self.input_dlta/self.input_lmbda),dim=(1,2,3))) * self.c1_lmbda
        self.conv1.bias = nn.Parameter(b_,requires_grad=False)
        
        w_ = self.conv2_beforenorm_weight.detach() * (self.c2_lmbda.detach()/self.c1_lmbda.detach())
        self.conv2.weight = nn.Parameter(w_,requires_grad=False)
        b_ = (self.conv2_beforenorm_bias.detach() - torch.sum(self.conv2_beforenorm_weight.detach() * (self.c1_dlta/self.c1_lmbda),dim=(1,2))) * self.c2_lmbda
        self.conv2.bias = nn.Parameter(b_,requires_grad=False)
     
        w_ = self.conv3_beforenorm_weight.detach() * (self.c3_lmbda.detach()/self.c2_lmbda.detach())
        self.conv3.weight = nn.Parameter(w_,requires_grad=False)
        b_ = (self.conv3_beforenorm_bias.detach() - torch.sum(self.conv3_beforenorm_weight.detach() * (self.c2_dlta/self.c2_lmbda),dim=(1,2))) * self.c3_lmbda
        self.conv3.bias = nn.Parameter(b_,requires_grad=False)
        
        w_ = self.conv4_beforenorm_weight.detach() * (self.c4_lmbda.detach()/self.c3_lmbda.detach())
        self.conv4.weight = nn.Parameter(w_,requires_grad=False)
        b_ = (self.conv4_beforenorm_bias.detach() - torch.sum(self.conv4_beforenorm_weight.detach() * (self.c3_dlta/self.c3_lmbda),dim=(1,2))) * self.c4_lmbda
        self.conv4.bias = nn.Parameter(b_,requires_grad=False)

        w_ = self.fc_beforenorm_weight.detach() * (self.fc_lmbda.detach()/self.c4_lmbda.detach())
        self.fc.weight = nn.Parameter(w_,requires_grad=False)
        b_ = (self.fc_beforenorm_bias.detach() - torch.sum(self.fc_beforenorm_weight.detach() * (self.c4_dlta/self.c4_lmbda),dim=(1))) * self.fc_lmbda
        self.fc.bias = nn.Parameter(b_,requires_grad=False)
        
        self.normed = True
        
        return
        
    @torch.no_grad()
    def calculate_norm_values(self, x):
        up = self.norm_max_fcn(x)
        lo = self.norm_min_fcn(x)
        self.input_lmbda = 1/(up-lo)
        self.input_dlta = -lo*self.input_lmbda
        #self.input_lmbda = self.input_lmbda.detach()
        #self.input_dlta = self.input_dlta.detach()
        
        x = self.conv1(x)
        
        up = self.norm_max_fcn(x)
        lo = self.norm_min_fcn(x)
        self.c1_lmbda = 1/up #1/(up-lo)
        self.c1_dlta = 0 #-lo*self.c1_lmbda
        #self.c1_lmbda = self.c1_lmbda.detach()
        #self.c1_dlta = self.c1_dlta.detach()
        
        x = self.conv1_bn(x)
        x = F.relu(x)

        x = x.squeeze(2)
        x = self.pool1(x)

        # conv2 block
        x = self.conv2(x)
        
        up = self.norm_max_fcn(x)
        lo = self.norm_min_fcn(x)
        self.c2_lmbda = 1/up #1/(up-lo)
        self.c2_dlta = 0 #-lo*self.c2_lmbda
        #self.c2_lmbda = self.c2_lmbda.detach()
        #self.c2_dlta = self.c2_dlta.detach()
        
        x = self.conv2_bn(x)
        x = F.relu(x)

        x = self.pool1(x)

        # conv3 block
        x = self.conv3(x)
        
        up = self.norm_max_fcn(x)
        lo = self.norm_min_fcn(x)
        self.c3_lmbda = 1/up #1/(up-lo)
        self.c3_dlta = 0 #-lo*self.c3_lmbda
        #self.c3_lmbda = self.c3_lmbda.detach()
        #self.c3_dlta = self.c3_dlta.detach()
        
        x = self.conv3_bn(x)
        x = F.relu(x)

        x = self.pool1(x)
        x = self.dropout(x)

        # conv4 block
        x = self.conv4(x)
        
        up = self.norm_max_fcn(x)
        lo = self.norm_min_fcn(x)
        self.c4_lmbda = 1/up #1/(up-lo)
        self.c4_dlta = 0 #-lo*self.c4_lmbda
        #self.c4_lmbda = self.c4_lmbda.detach()
        #self.c4_dlta = self.c4_dlta.detach()
        
        x = self.conv4_bn(x)
        x = F.relu(x)

        x = self.pool2(x)
        x = self.dropout(x)

        # fc block
        x = x.view(-1, 11 * 35 * 2)
        x = self.fc(x)
        
        up = self.norm_max_fcn(x)
        lo = self.norm_min_fcn(x)
        self.fc_lmbda = 1/up #1/(up-lo)
        self.fc_dlta = 0 #-lo*self.fc_lmbda
        #self.fc_lmbda = self.fc_lmbda.detach()
        #self.fc_dlta = self.fc_dlta.detach()

        return

class CINC7Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, hdf5_file, train=True, transform=None):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_raw = HDF5Dataset(hdf5_file, True)
        self.transform = transform
        
        #self.fold_idx = [0,1449,3011,4940,6750]
        self.fold_idx = [0,1706,3411,5117,6822]#8528
        
        self.val_idx = 4
        
        self.train=train

    def __len__(self):
        if self.val_idx == 4:
            len_val_fold = len(self.data_raw)-self.fold_idx[self.val_idx]
        else:
            len_val_fold = self.fold_idx[(self.val_idx+1)]-self.fold_idx[self.val_idx]
        
        if self.train:
            return (len(self.data_raw)-len_val_fold)
        else:
            return len_val_fold

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if isinstance(idx, slice):
            idx = range(*idx.indices(len(self.data_raw)))
        
        if isinstance(idx, int):
            idx = [idx]        
            
        ecgs = torch.empty(len(idx),18000)
        lbls = torch.empty(len(idx))
        
        for jj, ii in enumerate(idx):
            if self.train:
                if ii >= self.fold_idx[self.val_idx]:
                    if (self.val_idx+1<len(self.fold_idx)):
                        ii += self.fold_idx[self.val_idx+1]
            else:
                ii += self.fold_idx[self.val_idx]
            
            data = self.data_raw[ii]
            if len(data[0][0]) >= 18000:
                data_in = data[0][:,0:18000]
            else:
                data_in = torch.zeros((1,18000))
                data_in[0,0:len(data[0][0])] = data[0]
                
            ecgs[jj,:] = data_in
            lbls[jj] = data[1]
 
        sample = {'data_in': ecgs, 'label': lbls}

        if self.transform:
            sample = self.transform(sample)

        return sample

class DWT(object):
    """Calculate DWT
    """

    def __init__(self, dwt_lvl,squeeze_input=False):
        assert isinstance(dwt_lvl, int)
        self.dwt_lvl = dwt_lvl
        self.squeeze_input = squeeze_input

    def __call__(self, sample):
        data_in, lbls = sample['data_in'], sample['label']

        dwt_tmp = pywt.wavedec(data_in,'db2',level=self.dwt_lvl)
        dwts = torch.tensor(np.concatenate((dwt_tmp[0],dwt_tmp[1]),1).reshape((dwt_tmp[0].shape[0],2,1127)))

        if self.squeeze_input:
            dwts = dwts.squeeze(0)

        return {'data_in': dwts, 'label': lbls}
    
class DWTnNorm(object):
    """Calculate DWT
    """

    def __init__(self, dwt_lvl, lmbda, dlta,quant=False,squeeze_input=False,b=12):
        assert isinstance(dwt_lvl, int)
        self.dwt_lvl = dwt_lvl
        self.lmbda = lmbda
        self.dlta = dlta
        self.quant = quant
        self.b = b
        self.squeeze_input = squeeze_input
    
    def _q(self,x,b):
        # no saturation!
        result = round_halfup(x * 2**b) / 2**b
        return result 

    def __call__(self, sample):
        #pdb.set_trace()
        data_in, lbls = sample['data_in'], sample['label']

        if self.quant:
            db2wvlt = pywt.Wavelet('db2')
            dlo = self._q(torch.Tensor(db2wvlt.dec_lo),self.b)
            dhi = self._q(torch.Tensor(db2wvlt.dec_hi),self.b)
            din_q = self._q(data_in,self.b)
            
            if (len(din_q.shape) == 1):
                din_q = din_q.unsqueeze(0).unsqueeze(0)
            else:
                din_q = din_q.unsqueeze(1)    

            dcoeff = torch.cat((dlo.flip(0).unsqueeze(0),dhi.flip(0).unsqueeze(0)),0).unsqueeze(1)

            xin = din_q
            
            # due to change from padded sequence to valid conv the subsampling pattern deviates causing numerically different results
            # the following vector is intended to align the samples accordingly
            sbsmpl_start_idx = [0,1,0,0] 
            for ii in range(self.dwt_lvl):
                coeff = F.conv1d(xin,dcoeff,padding='valid')
                #pdb.set_trace()
                coeffs = coeff[:,:,sbsmpl_start_idx[ii]::2]
                coeffq = self._q(coeffs,self.b)
                
                xin = coeffq[:,0:1,:]
            
            #pdb.set_trace()
            #dwts_normed = coeffq * self.lmbda + self.dlta
            
            # since quantized version is not padded (no signal extension modes), the output samples are missing to trailing samples
            # these are extended with zeros to guarantee same alignment as unquantized version
            # in real-time usecase this does not matter, since it is one continuous sequence without edge cases
            dwts_normed = torch.cat((torch.zeros([1,2,2]),coeffq),2) * self.lmbda + self.dlta
            
        else:
            dwt_tmp = pywt.wavedec(data_in,'db2',level=self.dwt_lvl)
            dwts = torch.tensor(np.concatenate((dwt_tmp[0],dwt_tmp[1]),1).reshape((dwt_tmp[0].shape[0],2,1127)))
            dwts_normed = dwts * self.lmbda + self.dlta

        if self.squeeze_input:
            dwts_normed = dwts_normed.squeeze(0)

        return {'data_in': dwts_normed , 'label': lbls}
    
def split_into_5folds(label_sel):
    k=5
    num_Nor = 5076
    num_AF = 758
    num_Otr = 2415
    num_Nse = 279
    cnt = np.empty([k, 4])
    num = [math.ceil(num_Nor/k),math.ceil(num_AF/k),math.ceil(num_Otr/k),math.ceil(num_Nse/k)]

    vec_out = np.empty(len(label_sel))

    for j in range(k):
        cnt_Nor, cnt_AF, cnt_Otr, cnt_Nse = 4 * [0]
        no_Nor, no_AF, no_Otr, no_Nse = 4 * [0]

        for i in range(len(label_sel)):
            if label_sel[i] == 0:
                if num[0] * j <= cnt_Nor and cnt_Nor < num[0] * (j + 1):
                    vec_out[i] = j
                    no_Nor += 1
                cnt_Nor += 1

            elif label_sel[i] == 1:
                if num[1] * j <= cnt_AF and cnt_AF < num[1] * (j + 1):
                    vec_out[i] = j
                    no_AF += 1
                cnt_AF += 1

            elif label_sel[i] == 2:
                if num[2] * j <= cnt_Otr and cnt_Otr < num[2] * (j + 1):
                    vec_out[i] = j
                    no_Otr += 1
                cnt_Otr += 1

            elif label_sel[i] == 3:
                if num[3] * j <= cnt_Nse and cnt_Nse < num[3] * (j + 1):
                    vec_out[i] = j
                    no_Nse += 1
                cnt_Nse += 1

        cnt[j] = [no_Nor, no_AF, no_Otr, no_Nse]

    return vec_out

def get_f1(model, dat_in, lbl_in):
    cnn_output = model(dat_in)
    target = lbl_in
    _, pred = torch.max(cnn_output,1)

    total = target.size(0)
    correct = (pred == target).sum().item()

    # calc classes true positives, false positives and false negatives
    tpn = ((target == 0) & (pred == 0)).sum()
    fpn = ((target != 0) & (pred == 0)).sum()
    fnn = ((target == 0) & (pred != 0)).sum()

    tpa = ((target == 1) & (pred == 1)).sum()
    fpa = ((target != 1) & (pred == 1)).sum()
    fna = ((target == 1) & (pred != 1)).sum()

    tpo = ((target == 2) & (pred == 2)).sum()
    fpo = ((target != 2) & (pred == 2)).sum()
    fno = ((target == 2) & (pred != 2)).sum()

    F1n = (2*tpn.float())/(2*tpn.float()+fnn.float()+fpn.float())
    F1a = (2*tpa.float())/(2*tpa.float()+fna.float()+fpa.float())
    F1o = (2*tpo.float())/(2*tpo.float()+fno.float()+fpo.float())
    F1 = (F1n + F1a + F1o) / 3

    print(f'Total: {F1}, N: {F1n}, AF: {F1a}, O: {F1o}')
    return
    
def eval_model(model, dl, print_first=True, squeeze_input=False):
    correct = 0
    total = 0
    tpn,fpn,fnn,tpa,fpa,fna,tpo,fpo,fno = 9 * [0]
    pred_all = torch.empty(0,dtype=torch.long)
    for ii, dtrain in enumerate(tqdm(dl)):
        if squeeze_input:
            output = model(dtrain['data_in'].squeeze(1))
        else:
            output = model(dtrain['data_in'])
            
        if ((ii == 0) and (print_first==True)):
            print(output)

        target = dtrain['label'].flatten().long()
        _, pred = torch.max(output,1)
        pred_all = torch.cat((pred_all,pred))

        #pdb.set_trace()

        total += target.size(0)
        correct += (pred == target).sum().item()

        # calc classes true positives, false positives and false negatives
        tpn += ((target == 0) & (pred == 0)).sum()
        fpn += ((target != 0) & (pred == 0)).sum()
        fnn += ((target == 0) & (pred != 0)).sum()

        tpa += ((target == 1) & (pred == 1)).sum()
        fpa += ((target != 1) & (pred == 1)).sum()
        fna += ((target == 1) & (pred != 1)).sum()

        tpo += ((target == 2) & (pred == 2)).sum()
        fpo += ((target != 2) & (pred == 2)).sum()
        fno += ((target == 2) & (pred != 2)).sum()

    F1n = (2*tpn.float())/(2*tpn.float()+fnn.float()+fpn.float())
    F1a = (2*tpa.float())/(2*tpa.float()+fna.float()+fpa.float())
    F1o = (2*tpo.float())/(2*tpo.float()+fno.float()+fpo.float())
    F1 = (F1n + F1a + F1o) / 3
    Acc = correct / total

    print(f'Total: {F1}, N: {F1n}, AF: {F1a}, O: {F1o}, ACC: {Acc}')
    return F1, F1n, F1a, F1o, pred_all