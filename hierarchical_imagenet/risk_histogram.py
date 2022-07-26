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
from core import get_lhat
from ntree import *
import copy
import pdb

def subtree_sum_scores(memo, i, st, score, name_dict): # one at a time; score is a vector. i is for memoization
    if memo != None  and  st.name + str(i) in memo:
        return memo[st.name + str(i)]
    else:
        # else
        sum_scores = score[st.index] if st.index >= 0 else 0 
        for child in st.children:
            sum_scores += subtree_sum_scores(memo, i, child, score, name_dict)
        if memo != None:
            memo[st.name + str(i)] = sum_scores # add to cache
        return sum_scores

def hierarchical_loss(st, labels, idx_dict, name_dict):
    B = getMaxDepth(name_dict[idx_dict[0].parents[0]], idx_dict, name_dict)
    dists = np.zeros((len(st),))
    l_node = [copy.deepcopy(idx_dict[int(l)]) for l in labels.numpy()]
    for i in range(len(st)):
        dists[i] = getSubtreeLeafDistance(st[i],l_node[i])/B
    return dists

def get_heights(st, scores, labels, idx_dict, name_dict):
    heights = np.zeros((len(st),)) 
    starting_nodes = scores.argmax(dim=1)
    for i in range(len(st)):
        st_leaf = idx_dict[starting_nodes[i].item()]
        heights[i] = len(st_leaf.parents) - len(st[i].parents) 
    return heights

def get_subtree(scores, lam, idx_dict, name_dict, memo):
    start = torch.argmax(scores, dim=1).numpy() 
    st = [copy.deepcopy(idx_dict[s]) for s in start] # subtrees

    for i in range(start.shape[0]):
        parent_index = 0
        curr_sos = subtree_sum_scores(memo, i, name_dict[st[i].parents[parent_index]], scores[i], name_dict)
        if (i % 100) == 0:
            print(f'{i}\r', end='')
        while parent_index < len(st[i].parents) and curr_sos > lam:  # every iterate through the loop, the set shrinks.
            parent_index += 1
            curr_sos = subtree_sum_scores(memo, i, name_dict[st[i].parents[min(parent_index,len(st[i].parents)-1)]], scores[i], name_dict) 
        st[i] = name_dict[st[i].parents[min(parent_index,len(st[i].parents)-1)]]
    return st
    
def trial_precomputed(example_loss_table, example_height_table, lambdas_example_table, alpha, num_lam, num_calib, batch_size):
    total=example_loss_table.shape[0]
    perm = torch.randperm(example_loss_table.shape[0])
    example_loss_table = example_loss_table[perm]
    example_height_table = example_height_table[perm]
    calib_losses, val_losses = (example_loss_table[0:num_calib], example_loss_table[num_calib:])
    calib_heights, val_heights = (example_height_table[0:num_calib], example_height_table[num_calib:])

    lhat = get_lhat(calib_losses[:,::-1], lambdas_example_table[::-1], alpha)

    losses = val_losses[:,np.argmax(lambdas_example_table == lhat)]
    heights = val_heights[:,np.argmax(lambdas_example_table == lhat)]

    return losses.mean(), np.random.choice(heights), lhat

def plot_histograms(df,alpha):
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))

    minrisk = df['risk'].min()
    maxrisk = df['risk'].max()

    risk_bins = np.arange(minrisk, maxrisk, 0.0005) 
    
    axs[0].hist(df['risk'].to_numpy(), risk_bins, alpha=0.7, density=True)

    # Sizes will be 10 times as big as risk, since we pool it over runs.
    sizes = df['height'].to_numpy()
    d = np.diff(np.unique(sizes)).min()
    lofb = sizes.min() - float(d)/2
    rolb = sizes.max() + float(d)/2
    axs[1].hist(sizes, np.arange(lofb,rolb+d, d), alpha=0.7, density=True)
    
    axs[0].set_xlabel('risk')
    axs[0].locator_params(axis='x', nbins=8)
    axs[0].set_ylabel('density')
    axs[0].set_yticks([0,100])
    axs[0].axvline(x=alpha,c='#999999',linestyle='--',alpha=0.7)
    axs[1].set_xlabel('height')
    axs[1].set_yscale('log')
    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    axs[1].set_xlim([-0.5,rolb])
    #axs[1].legend()
    plt.tight_layout()
    os.makedirs('outputs/histograms/',exist_ok=True)
    plt.savefig( (f'outputs/histograms/{alpha}_{num_calib}_hierarchical_imagenet_histograms').replace('.','_') + '.pdf')
    print(f"The mean and standard deviation of the risk over {len(df)} trials are {df['risk'].mean()} and {df['risk'].std()} respectively.")

def load_imagenet_tree():
    with open('./wordnet_hierarchy.json', 'r') as file:
        data = file.read()
    imagenet_dict = json.loads(data)
    t = dict2tree(imagenet_dict)
    idx_dict = getIndexDict(t)
    name_dict = getNameDict(t)
    return idx_dict, name_dict

def get_example_loss_and_height_tables(scores, labels, idx_dict, name_dict, lambdas_example_table, num_calib):
    lam_len = len(lambdas_example_table)
    lam_low = min(lambdas_example_table)
    lam_high = max(lambdas_example_table)
    fname_loss = f'./.cache/{lam_low}_{lam_high}_{lam_len}_example_loss_table.npy'
    fname_height = f'./.cache/{lam_low}_{lam_high}_{lam_len}_example_height_table.npy'
    try:
        loss_table = np.load(fname_loss)
        height_table = np.load(fname_height)
    except:
        loss_table = np.zeros((scores.shape[0], lam_len))
        height_table = np.zeros((scores.shape[0], lam_len))
        memo = {}
        for j in range(lam_len):
            sts = get_subtree(scores, lambdas_example_table[j], idx_dict, name_dict, memo)
            losses_lam = hierarchical_loss(sts,labels,idx_dict,name_dict)
            loss_table[:,j] = losses_lam
            height_table[:,j] = get_heights(sts, scores, labels, idx_dict, name_dict)

        np.save(fname_loss, loss_table)
        np.save(fname_height, height_table)

    return loss_table, height_table

def experiment(losses,alpha,lambdas_example_table,num_lam,num_calib,num_trials,imagenet_val_dir,batch_size=128):
        fname = f'.cache/{alpha}_{num_lam}_{num_calib}_{num_trials}_hierarchical_dataframe.pkl'

        df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","risk","height","alpha"])
        try:
            df = pd.read_pickle(fname)
        except FileNotFoundError:
            dataset_precomputed = get_logits_dataset('ResNet152', 'Imagenet', imagenet_val_dir)
            print('Dataset loaded')
            
            classes_array = get_imagenet_classes()
            T = platt_logits(dataset_precomputed)
            
            logits, labels = dataset_precomputed.tensors
            scores = (logits/T.cpu()).softmax(dim=1)

            idx_dict, name_dict = load_imagenet_tree()

            with torch.no_grad():
                # get the precomputed binary search
                example_loss_table, example_height_table = get_example_loss_and_height_tables(scores, labels, idx_dict, name_dict, lambdas_example_table, num_calib)

                local_df_list = []
                for i in tqdm(range(num_trials)):
                    risk, height, lhat = trial_precomputed(example_loss_table, example_height_table, lambdas_example_table, alpha, num_lam, num_calib, batch_size)
                    dict_local = {
                                    "$\\hat{\\lambda}$": lhat,
                                    "risk": risk,
                                    "height": height,
                                    "alpha": alpha 
                                 }
                    df_local = pd.DataFrame(dict_local, index=[i])
                    local_df_list += [df_local]
                df = pd.concat(local_df_list, axis=0, ignore_index=True)
                df.to_pickle(fname)
        plot_histograms(df,alpha)

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
    imagenet_val_dir = '~/data/ilsvrc/val/'
    os.makedirs('./.cache', exist_ok=True)

    losses = torch.ones((1000,))
    alphas = [0.02, 0.05]
    num_lam = 1500 
    num_calib = 30000 
    num_trials = 1000 
    lambdas_example_table = np.linspace(0,1,1000)
    
    for alpha in alphas:
        print(f"\n\n\n ============           NEW EXPERIMENT alpha={alpha}           ============ \n\n\n") 
        experiment(losses,alpha,lambdas_example_table,num_lam,num_calib,num_trials,imagenet_val_dir)
