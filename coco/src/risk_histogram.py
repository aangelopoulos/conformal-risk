import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
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
from core import get_lhat
import seaborn as sns
from ml_decoder.helper_functions.bn_fusion import fuse_bn_recursively
from ml_decoder.models import create_model
from ml_decoder.models.tresnet.tresnet import InplacABN_to_ABN
import pdb

parser = argparse.ArgumentParser(description='PyTorch MS_COCO infer')
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--model-path', type=str, default='../models/tresnet_xl_COCO_640_91_4.pth')
parser.add_argument('--pic-path', type=str, default='./pics/dog.jpg')
parser.add_argument('--model-name', type=str, default='tresnet_xl')
parser.add_argument('--input-size', type=int, default=640)
# parser.add_argument('--dataset-type', type=str, default='MS-COCO')
parser.add_argument('--th', type=float, default=0.75)
parser.add_argument('--top-k', type=float, default=80)
# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)

def get_example_loss_and_size_tables(scores, labels, lambdas_example_table, num_calib):
    lam_len = len(lambdas_example_table)
    lam_low = min(lambdas_example_table)
    lam_high = max(lambdas_example_table)
    fname_loss = f'../.cache/{lam_low}_{lam_high}_{lam_len}_example_loss_table.npy'
    fname_sizes = f'../.cache/{lam_low}_{lam_high}_{lam_len}_example_size_table.npy'
    try:
        loss_table = np.load(fname_loss)
        sizes_table = np.load(fname_sizes)
    except:
        loss_table = np.zeros((scores.shape[0], lam_len))
        sizes_table = np.zeros((scores.shape[0], lam_len))
        print("caching loss and size tables")
        for j in tqdm(range(lam_len)):
            est_labels = scores >= lambdas_example_table[j]
            loss, sizes = get_metrics_precomputed(est_labels, labels)
            loss_table[:,j] = loss 
            sizes_table[:,j] = sizes

        np.save(fname_loss, loss_table)
        np.save(fname_sizes, sizes_table)

    return loss_table, sizes_table

def trial_precomputed(example_loss_table, example_size_table, lambdas_example_table, alpha, num_lam, num_calib, batch_size):
    rng_state = np.random.get_state()
    np.random.shuffle(example_loss_table)
    np.random.set_state(rng_state)
    np.random.shuffle(example_size_table)

    calib_losses, val_losses = (example_loss_table[0:num_calib], example_loss_table[num_calib:])
    calib_sizes, val_sizes = (example_size_table[0:num_calib], example_size_table[num_calib:])

    lhat = get_lhat(calib_losses, lambdas_example_table, alpha)

    val_losses_lhat = val_losses[:,np.argmax(lambdas_example_table == lhat)]
    val_sizes_lhat = val_sizes[:,np.argmax(lambdas_example_table == lhat)]

    return val_losses_lhat.mean(), torch.tensor(val_sizes_lhat), lhat

def plot_histograms(df,alpha):
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))

    minrisk = df['risk'].min()
    maxrisk = df['risk'].max()

    risk_bins = np.arange(minrisk, maxrisk, 0.001) 
    size = df['size'].to_numpy()
    d = np.diff(np.unique(size)).min()
    lofb = size.min() - float(d)/2
    rolb = size.max() + float(d)/2
    size_bins = np.arange(lofb,rolb+d, d)

    axs[0].hist(df['risk'], risk_bins, alpha=0.7, density=True)

    axs[1].hist(size, size_bins, alpha=0.7, density=True)
    
    axs[0].set_xlabel('risk')
    axs[0].locator_params(axis='x', nbins=4)
    axs[0].set_ylabel('density')
    #axs[0].set_yticks([0,100])
    axs[0].axvline(x=alpha,c='#999999',linestyle='--',alpha=0.7)
    axs[1].set_xlabel('size')
    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    #axs[1].legend()
    plt.tight_layout()
    os.makedirs('../outputs/histograms/', exist_ok=True)
    plt.savefig('../' + (f'outputs/histograms/{alpha}_coco_histograms').replace('.','_') + '.pdf')
    print(f"Average threshold: ", df["$\\hat{\\lambda}$"].mean())
    print(f"The mean and standard deviation of the risk over {len(df)} trials are {df['risk'].mean()} and {df['risk'].std()} respectively.")

def experiment(alpha,num_lam,num_calib,lambdas_example_table,num_trials,batch_size,coco_val_2017_directory,coco_instances_val_2017_json):
    fname = f'../.cache/{alpha}_{num_calib}_{num_lam}_{num_trials}_dataframe.pkl'
    os.makedirs('../.cache', exist_ok=True)

    df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","risk","size","alpha"])
    try:
        df = pd.read_pickle(fname)
    except FileNotFoundError:
        # parsing args
        args = parser.parse_args()

        # dataset

        dataset = tv.datasets.CocoDetection(coco_val_2017_directory,coco_instances_val_2017_json,transform=tv.transforms.Compose([tv.transforms.Resize((args.input_size, args.input_size)),
                                                                                                                                                         tv.transforms.ToTensor()]))
        print('Dataset loaded')
        
        # Setup model
        print('creating model {}...'.format(args.model_name))
        model = create_model(args, load_head=True).cuda()
        state = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state['model'], strict=True)
        ########### eliminate BN for faster inference ###########
        model = model.cpu()
        model = InplacABN_to_ABN(model)
        model = fuse_bn_recursively(model)
        model = model.cuda().eval()
        state = torch.load(args.model_path, map_location='cpu')
        classes_list = np.array(list(state['idx_to_class'].values()))
        args.num_classes = state['num_classes']
        model.eval()
        print('Model Loaded')
        corr = get_correspondence(classes_list,dataset.coco.cats)

        # get dataset
        dataset_fname = '../.cache/coco_val.pkl'
        if os.path.exists(dataset_fname):
            dataset_precomputed = pkl.load(open(dataset_fname,'rb'))
            print(f"Precomputed dataset loaded. Size: {len(dataset_precomputed)}")
        else:
            dataset_precomputed = get_scores_targets(model, torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True), corr)
            pkl.dump(dataset_precomputed,open(dataset_fname,'wb'),protocol=pkl.HIGHEST_PROTOCOL)
        scores, labels = dataset_precomputed.tensors

        # get the loss and size table 
        example_loss_table, example_size_table = get_example_loss_and_size_tables(scores, labels, lambdas_example_table, num_calib)
        
        local_df_list = []
        for i in tqdm(range(num_trials)):
            risk, sizes, lhat = trial_precomputed(example_loss_table, example_size_table, lambdas_example_table, alpha, num_lam, num_calib, batch_size)
            dict_local = {"$\\hat{\\lambda}$": lhat,
                            "risk": risk,
                            "size": np.random.choice(sizes),
                            "alpha": alpha
                         }
            df_local = pd.DataFrame(dict_local, index=[i])
            local_df_list = local_df_list + [df_local]
        df = pd.concat(local_df_list, axis=0, ignore_index=True)
        df.to_pickle(fname)

    plot_histograms(df,alpha)


if __name__ == "__main__":
    with torch.no_grad():
        sns.set(palette='pastel',font='serif')
        sns.set_style('white')
        fix_randomness(seed=0)
        coco_val_2017_directory = '../data/val2017'
        coco_instances_val_2017_json = '../data/annotations_trainval2017/instances_val2017.json'

        alphas = [0.1,0.05]
        num_lam = 1500 
        num_calib = 2000 
        num_trials = 1000
        batch_size = 100
        lambdas_example_table = np.linspace(0,1,num_lam)
        
        for alpha in alphas:
            print(f"\n\n\n ============           NEW EXPERIMENT alpha={alpha}           ============ \n\n\n") 
            experiment(alpha,num_lam,num_calib,lambdas_example_table,num_trials,batch_size,coco_val_2017_directory,coco_instances_val_2017_json)
