import numpy as np
import torch
import os
import wfdb

import pandas as pd
import pdb

def download_wfdb_database(root_str,db_name):
    dl_path = root_str + 'data/' + db_name
    if os.path.exists(dl_path):
        print('dataset folder exists, skip db: ' + db_name + ' ...')
    else:
        wfdb.dl_database(db_name,dl_path,keep_subdirs=False)
        
def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("use cuda")
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')

    if args.download_dataset:
        download_wfdb_database(args.repo_source,'afdb')

    # list of indices of files
    lst = [item for item in os.listdir(args.repo_source+'data/'+'afdb') if 'dat' in item]
    lst = list(sorted(set([it[:-4] for it in lst])))

    # data and labels
    npz_filename = args.repo_source+'data/'+'afdb/'+'dat.npz'
    if os.path.exists(npz_filename):
        print(f'load: {npz_filename}')
        npzfile = np.load(npz_filename)
        x = npzfile['arr_0']
        y = npzfile['arr_1']
        patidx = npzfile['arr_2']
    else:
        seq_len = 5000 #360
        x = np.empty((0,2,seq_len),dtype='int32')
        y = np.empty(0,dtype='int64')
        patidx = np.empty(0,dtype='int64')
        for it in lst:
            print('reading: ' + it)
            rec = wfdb.rdrecord(args.repo_source+'data/'+'afdb/'+it)
            ann = wfdb.rdann(args.repo_source+'data/'+'afdb/'+it,extension='atr')
    
            # for idx_lbl, it_lbl_time in enumerate(ann.sample):
            #     # remove samples at the edge of sequence
            #     if it_lbl_time < seq_len/2 or it_lbl_time > (len(rec.p_signal)-seq_len/2):
            #         continue

            #     dat = np.expand_dims(rec.p_signal.T.astype('float32')[:,int(it_lbl_time-seq_len/2):int(it_lbl_time+seq_len/2)],0)
            #     lbl = [ord(ii) for ii in ann.symbol[idx_lbl]]
                
            #     x = np.append(x,dat,axis=0)
            #     y = np.append(y,lbl,axis=0)

            annidx = 0
            for ii in range(0,len(rec.p_signal)-seq_len,seq_len):
               
                dat = np.expand_dims(rec.p_signal.T.astype('float32')[:,ii:ii+seq_len],0)
                lbl = [0 if ann.aux_note[annidx][1:] == 'N' else 1 if ann.aux_note[annidx][1:] == 'AFIB' else 2 if ann.aux_note[annidx][1:] == 'AFL' else 3]

                x = np.append(x,dat,axis=0)
                y = np.append(y,lbl,axis=0)

                if annidx < len(ann.aux_note)-1:
                    if ii+seq_len > ann.sample[annidx+1]:
                        annidx += 1

            patidx = np.append(patidx,np.ones(y.shape[0]-patidx.shape[0],dtype='int64')*int(it),axis=0)
            #pdb.set_trace()
            

        np.savez(npz_filename,x,y,patidx)

    # subselect first channel and classes N,AFIB only
    
    x = x[:,:1,:] #chn_in

    idx_sel = np.logical_or(y==0,y==1) #classes
    x = x[idx_sel,:,:]
    y = y[idx_sel]
    patidx = patidx[idx_sel]
   
    return (device, x, y, patidx)