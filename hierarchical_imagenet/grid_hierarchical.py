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
from risk_histogram import get_subtree, hierarchical_loss, load_imagenet_tree, platt_logits 

def get_grid_example_loss_and_height_tables(lambdas_example_table):
    lam_len = len(lambdas_example_table)
    lam_low = min(lambdas_example_table)
    lam_high = max(lambdas_example_table)
    fname_loss = f'./.cache/{lam_low}_{lam_high}_{lam_len}_example_loss_table.npy'
    fname_height = f'./.cache/{lam_low}_{lam_high}_{lam_len}_example_height_table.npy'
    loss_table = np.load(fname_loss)
    height_table = np.load(fname_height)
    return loss_table, height_table

def generate_plot(lambdas_example_table,alpha,num_lam,num_calib,batch_size=128):
    fname_imgs_to_plot = './.cache/imgs_to_plot.pkl'
    fname_hier_to_plot = './.cache/hier_to_plot.pkl'
    fname_true_strings_to_plot = './.cache/true_strings_to_plot.pkl'
    num_plot = 12 
    try:
        imgs_to_plot = pkl.load( open(fname_imgs_to_plot, 'rb') )
        hier_to_plot = pkl.load( open(fname_hier_to_plot, 'rb') )
        true_strings_to_plot = pkl.load( open(fname_true_strings_to_plot, 'rb') )
    except:
        with torch.no_grad():
            # load imagenet hierarchy structure 
            idx_dict, name_dict = load_imagenet_tree()

            # Get the proper threshold
            loss_table, _ = get_grid_example_loss_and_height_tables(lambdas_example_table)
            np.random.shuffle(loss_table)
            calib_loss_table = loss_table[0:num_calib] 

            # calibrate
            lhat = get_lhat(calib_loss_table[:,::-1], lambdas_example_table[::-1], alpha)

            # validate
            memo = None # no more memo.
            model = get_model('ResNet18')

            transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std =[0.229, 0.224, 0.225])
                            ])
            
            num_test = 10000 
            dataset = torchvision.datasets.ImageFolder('~/data/ilsvrc/val/', transform)

            memo = None 

            imgs = []
            val_scores = torch.zeros((num_test,1000))
            val_labels = torch.zeros((num_test,))
            rand_idxs = torch.randint(low=0, high=len(dataset), size=(num_test,))
            for i in range(num_test):
                ri = rand_idxs[i]
                img, lab = dataset[ri]
                imgs = imgs + [img]
                val_scores[i] = model(img[None,:]).softmax(dim=1)
                val_labels[i] = lab

            st = get_subtree(val_scores, lhat, idx_dict, name_dict, memo)

            losses = hierarchical_loss(st, val_labels, idx_dict, name_dict)

            heights = torch.tensor(np.array([len(s.children) for s in st]))

            top1 = torch.argmax(val_scores.softmax(dim=1),dim=1)
            top1_strings = [idx_dict[t.item()].name.split(',')[0] for t in top1]
            hier_strings = [s.name.split(',')[0] for s in st]
            true_strings = [idx_dict[v.item()].name.split(',')[0] for v in val_labels]
            correct = losses == 0
            top1_correct = top1 == val_labels

            imgs_to_plot = []
            top1_to_plot = []
            hier_to_plot = []
            true_strings_to_plot = []
            k = 0
            # Places where ResNet18 is wrong are fairly rare, so we need to find them.
            for i in range(len(top1_strings)):
                if correct[i] and not top1_correct[i] and k < num_plot:
                    img_to_plot = (imgs[i] * torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)) + torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1)
                    imgs_to_plot = imgs_to_plot + [img_to_plot.permute(1,2,0)]
                    top1_to_plot = top1_to_plot + [[top1_strings[i], true_strings[i]]]
                    hier_to_plot = hier_to_plot + [[hier_strings[i], true_strings[i]]]
                    true_strings_to_plot = true_strings_to_plot + [[true_strings[i], top1_strings[i]]]
                    k = k + 1
                    print(f"true: {true_strings[i]}, hier: {hier_strings[i]}, top1: {top1_strings[i]}, correct: {correct[i]}, top1_correct: {top1_correct[i]}")
            pkl.dump( imgs_to_plot, open(fname_imgs_to_plot, 'wb'))
            pkl.dump( hier_to_plot, open(fname_hier_to_plot, 'wb'))
            pkl.dump( true_strings_to_plot, open(fname_true_strings_to_plot, 'wb'))

    order = [0,1,2,4,5,6,7,9]
    imgs_to_plot_ordered = [imgs_to_plot[o] for o in order]
    hier_to_plot_ordered = [hier_to_plot[o] for o in order]
    true_strings_to_plot_ordered = [true_strings_to_plot[o] for o in order] 

    gridplot_imgs_tree(imgs_to_plot_ordered, hier_to_plot_ordered, true_strings_to_plot_ordered, 2, int(np.ceil(num_plot/2)))

def gridplot_imgs_tree(imgs,est_labels,labels,rows,cols):
    fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(cols*10,rows*10))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    for idx, img in enumerate(imgs):
        r = idx//4
        c = idx % 4
        axs[r,c].axis('off')
        axs[r,c].imshow(img, aspect='equal')
        # Remove last newline
        hier_labelstr = est_labels[idx][0]
        top1_labelstr = labels[idx][1] 
        corr_labelstr =  labels[idx][0] 
        all_labelstr = corr_labelstr + '\n' + hier_labelstr + '\n' + top1_labelstr   
        bluetext = corr_labelstr + '\n' + hier_labelstr 

        # Resize text
        fontsize = 64

        # Make a fake bbox first.
        axs[r,c].text(0.05,0.95,all_labelstr,transform=axs[r,c].transAxes,fontsize=fontsize,color='#00000000',verticalalignment='top',bbox=props)
        axs[r,c].text(0.05,0.95,all_labelstr,transform=axs[r,c].transAxes,fontsize=fontsize,color='#ff4555',verticalalignment='top')
        axs[r,c].text(0.05,0.95,bluetext,transform=axs[r,c].transAxes,fontsize=fontsize,color='#40B5BC',verticalalignment='top')
        axs[r,c].text(0.05,0.95,corr_labelstr,transform=axs[r,c].transAxes,fontsize=fontsize,color='k',verticalalignment='top')

    for i in range(rows):
        for j in range(cols):
            axs[i,j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    plt.savefig('./outputs/hierarchical_grid_fig.pdf')

if __name__ == "__main__":
    sns.set(palette='pastel',font='serif')
    sns.set_style('white')
    fix_randomness(seed=0)

    losses = torch.ones((1000,))
    alphas = [0.05]
    num_lam = 100 
    num_calib = 40000 
    lambdas_example_table = np.linspace(0,1,1000)
    
    for alpha in alphas:
        print(f"\n\n\n ============           NEW EXPERIMENT alpha={alpha}           ============ \n\n\n") 
        generate_plot(lambdas_example_table,alpha,num_lam,num_calib)
