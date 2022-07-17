import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
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
from core.get_lhat import get_lhat_selective 
import pdb

def plot_results(df,alpha,top_scores,corrects,lambdas):
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))

    conditional_risk_curve_y = [ 1-corrects[ top_scores > 1-lam ].float().mean() for lam in lambdas ]
    conditional_risk_curve_x = [ 1-(top_scores > 1-lam).float().mean() for lam in lambdas ]
    axs[1].plot(conditional_risk_curve_x,conditional_risk_curve_y,color='k',linewidth=3,label='selective risk')

    risks = []
    labels = []

    sns.violinplot(data=df['selective risk'], ax=axs[0], orient='h', inner=None)
    
    axs[0].set_xlabel('selective risk')
    axs[0].locator_params(axis='x', nbins=4)
    axs[0].axvline(x=alpha,c='#999999',linestyle='--',alpha=0.7)
    axs[0].set_yticklabels(labels)
    axs[1].set_xlabel('abstention frequency')
    axs[1].axhline(y=alpha, c='#999999', linestyle=':',label="$\\alpha$", alpha=0.7)
    axs[1].legend(loc='upper right')
    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    plt.tight_layout()
    os.makedirs('outputs/histograms',exist_ok=True)
    plt.savefig((f'outputs/histograms/selective_risk_{alpha}_imagenet_histograms').replace('.','_') + '.pdf')
    print(f"The base accuracy of the model is {corrects.float().mean()}")
    print(f"Over {len(df)} trials, the average selective risk is {df['selective risk'].mean()} with a standard deviation of {df['selective risk'].std()}.")

def get_data(imagenet_val_dir):
    dataset_precomputed = get_logits_dataset('ResNet152', 'Imagenet', imagenet_val_dir)
    print('Dataset loaded')
    
    classes_array = get_imagenet_classes()
    T = platt_logits(dataset_precomputed)
    
    logits, labels = dataset_precomputed.tensors
    top_scores, top_classes = (logits/T.cpu()).softmax(dim=1).max(dim=1)
    corrects = top_classes==labels
    return top_scores, corrects

def trial_precomputed(top_scores, corrects, alpha, lambdas, num_calib):
    total=top_scores.shape[0]
    m=1000
    perm = torch.randperm(total)
    top_scores = top_scores[perm]
    corrects = corrects[perm].float()
    calib_scores, val_scores = (top_scores[0:num_calib], top_scores[num_calib:])
    calib_corrects, val_corrects = (corrects[0:num_calib], corrects[num_calib:])

    calib_scores, indexes = calib_scores.sort()
    calib_corrects = calib_corrects[indexes] 

    lhat = get_lhat_selective(1-corrects, top_scores, lambdas, alpha, B=1)

    val_predictions = val_scores > lhat

    risk = 1-val_corrects[val_predictions].float().mean()
    risk = np.nan_to_num(risk)
    
    fraction_abstentions = 1-val_predictions.float().mean()
    
    return risk, fraction_abstentions, lhat

def experiment(alpha,lambdas,num_calib,num_trials,imagenet_val_dir):
    fname = f'./.cache/{alpha}_{num_calib}_{num_trials}_dataframe.pkl'
    df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","selective risk","abstention frequency","alpha"])
    top_scores, corrects = get_data(imagenet_val_dir)
    
    try:
        df = pd.read_pickle(fname)
    except FileNotFoundError:

        with torch.no_grad():
            local_df_list = []
            for i in tqdm(range(num_trials)):
                risk, fraction_abstentions, lhat = trial_precomputed(top_scores, corrects, alpha, lambdas, num_calib)
                dict_local = {"$\\hat{\\lambda}$": lhat,
                                "selective risk": risk,
                                "abstention frequency": fraction_abstentions,
                                "alpha": alpha,
                                "index": [0]
                             }
                df_local = pd.DataFrame(dict_local)
                local_df_list = local_df_list + [df_local]
            df = pd.concat(local_df_list, axis=0, ignore_index=True)
            df.to_pickle(fname)

    plot_results(df, alpha, top_scores, corrects, lambdas)

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

    alphas = [0.15,0.1,0.05]
    num_trials = 1000 
    num_calib = 30000
    lambdas = np.linspace(0,1,1000)
    
    for alpha in alphas:
        print(f"\n\n\n ============           NEW EXPERIMENT alpha={alpha}          ============ \n\n\n") 
        experiment(alpha,lambdas,num_calib,num_trials,imagenet_val_dir)
