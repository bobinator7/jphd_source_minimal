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
        download_wfdb_database(args.repo_source,'challenge-2017/training')
        #wfdb.dl_files('challenge-2017', args.repo_source+'data/challenge-2017', ['training/RECORDS-af','training/RECORDS-noisy','training/RECORDS-normal','training/RECORDS-other'])
        wfdb.dl_files('challenge-2017', args.repo_source+'data/challenge-2017/training', ['REFERENCE-v3.csv'])
        #download_wfdb_database(args.repo_source,'challenge-2017/validation')
        #wfdb.dl_files('challenge-2017', args.repo_source+'data/challenge-2017', ['validation/RECORDS-af','validation/RECORDS-noisy','validation/RECORDS-normal','validation/RECORDS-other'])

    # list of indices of files
    lst = list(sorted(set([it[:-4] for it in os.listdir(args.repo_source+'data/'+'challenge-2017/training')])))
    lst = [item for item in lst if 'RE' not in item]
    

    # data
    recs = []
    max_len=18000
    for it in lst:
        print('reading: ' + it)
        rec = wfdb.rdrecord(args.repo_source+'data/'+'challenge-2017/training/'+it)
        dat_raw = np.expand_dims(rec.p_signal.T.astype('float32'),0)
        dat_pad = np.pad(dat_raw[:,:,:max_len], ((0,0),(0,0),(0,max_len-dat_raw[:,:,:max_len].shape[2])), 'constant', constant_values=((0,0),(0,0),(0,0)))
        recs.append(dat_pad)

    x = np.concatenate(recs,axis=0)

    # labels (0: normal, 1: af, 2: other, 3: noisy)
    print('read labels')
    y = np.empty(x.shape[0],dtype='int64')

    f = pd.read_csv(args.repo_source+'data/challenge-2017/training/REFERENCE-v3.csv',header=None)
    y[f.iloc[:,1]=='N'] = 0
    y[f.iloc[:,1]=='A'] = 1
    y[f.iloc[:,1]=='O'] = 2
    y[f.iloc[:,1]=='~'] = 3
        
    #pdb.set_trace()

    # with open(args.repo_source+'data/challenge-2017/training/RECORDS-normal') as f:
    #     lines = f.read().splitlines()
    #     idx = np.array([int(it[5:])-1 for it in lines])
    #     y[idx] = 0

    # with open(args.repo_source+'data/challenge-2017/training/RECORDS-af') as f:
    #     lines = f.read().splitlines()
    #     idx = np.array([int(it[5:])-1 for it in lines])
    #     y[idx] = 1

    # with open(args.repo_source+'data/challenge-2017/training/RECORDS-other') as f:
    #     lines = f.read().splitlines()
    #     idx = np.array([int(it[5:])-1 for it in lines])
    #     y[idx] = 2

    # with open(args.repo_source+'data/challenge-2017/training/RECORDS-noisy') as f:
    #     lines = f.read().splitlines()
    #     idx = np.array([int(it[5:])-1 for it in lines])
    #     y[idx] = 3

    return (device, x, y)