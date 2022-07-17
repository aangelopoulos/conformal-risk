import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import torch
import torchvision as tv
import argparse
import time
import numpy as np
from scipy.stats import binom
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from utils import *
import seaborn as sns
from core.uniform_concentration import *
from core.pfdr import *
import pdb

def get_lambdas_vs_pfdps_frac_predict(lambdas,imagenet_val_dir):
    dataset_precomputed = get_logits_dataset('ResNet152', 'Imagenet', imagenet_val_dir)
    print("Dataset loaded")
    classes_array = get_imagenet_classes()
    T = platt_logits(dataset_precomputed)
    
    logits, labels = dataset_precomputed.tensors
    top_scores, top_classes = (logits/T.cpu()).softmax(dim=1).max(dim=1)
    corrects = top_classes==labels

    with torch.no_grad():
        idxs = [ top_scores > lam for lam in lambdas ]
        frac_predict = np.array([ idx.float().mean() for idx in idxs ])
        pfdps = np.array([ 1-corrects[idx].float().mean() for idx in idxs ]) 
        pfdps = np.nan_to_num(pfdps)
        return pfdps, frac_predict

def platt_logits(calib_dataset, max_iters=10, lr=0.01, epsilon=0.01):
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=1024, shuffle=False, pin_memory=True) 
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T 

if __name__ == "__main__":
    sns.set(palette='pastel',font='serif')
    sns.set_style('white')
    fix_randomness(seed=0)

    imagenet_val_dir = '/scratch/group/ilsvrc/val' #TODO: Replace this with YOUR location of imagenet val set.

    lambdas = np.linspace(0,1,1000)
    
    pfdps, frac_predict = get_lambdas_vs_pfdps_frac_predict(lambdas,imagenet_val_dir)
    print("HI")
