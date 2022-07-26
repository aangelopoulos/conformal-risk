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
parser.add_argument('--th', type=float, default=0.7489)
parser.add_argument('--top-k', type=float, default=80)
# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)

def random_example(dataset, model, scores_to_labels, corr, classes_list):
    i = random.randint(0,len(dataset)-1) 
    img = dataset[i][0]
    ann = dataset[i][1][0]

    labels = []
    annotations = dataset.coco.getAnnIds(imgIds=int(ann['image_id'])) 
    for annotation in dataset.coco.loadAnns(annotations):
        labels = labels + [classes_list[corr[annotation['category_id']]]]
    labels = list(np.unique(np.array(labels)))
    est_labels = scores_to_labels(model(img.unsqueeze(0).cuda()).cpu())
    return img.permute(1,2,0), est_labels, [labels]

def gridplot_imgs(imgs,est_labels,labels,rows,cols):
    fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(cols*10,rows*10))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    for idx, img in enumerate(imgs):
        r = idx//cols
        c = idx % cols
        axs[r,c].axis('off')
        axs[r,c].imshow(img, aspect='equal')
        corr_labelstr = ""
        est_labelstr = ""
        all_labelstr = ""
        fake_labelstr = ""
        num_labels = 0
        for i in range(len(est_labels[idx])):
            if est_labels[idx][i] in labels[idx]:
                corr_labelstr += est_labels[idx][i] + '\n'
                est_labelstr = '\n' + est_labelstr
                all_labelstr = '\n' + all_labelstr 
                fake_labelstr += est_labels[idx][i] + '\n'
            else:
                est_labelstr += est_labels[idx][i] + '\n'
                all_labelstr += '\n'
                fake_labelstr += est_labels[idx][i] + '\n'
            num_labels += 1

        for i in range(len(labels[idx])):
            if labels[idx][i] not in est_labels[idx]:
                all_labelstr += labels[idx][i] + '\n'
                fake_labelstr += labels[idx][i] + '\n'
                num_labels += 1

        # Remove last newline
        fake_labelstr = fake_labelstr[0:-1]
        all_labelstr = all_labelstr[0:-1]
        est_labelstr = est_labelstr[0:-1]
        corr_labelstr = corr_labelstr[0:-1] 

        # Resize text
        fontsize = 32
        if(num_labels <= 5):
            fontsize = 48

        # Make a fake bbox first.
        axs[r,c].text(0.05,0.95,fake_labelstr,transform=axs[r,c].transAxes,fontsize=fontsize,color='#00000000',verticalalignment='top',bbox=props)
        axs[r,c].text(0.05,0.95,all_labelstr,transform=axs[r,c].transAxes,fontsize=fontsize,color='#ff4555',verticalalignment='top')
        axs[r,c].text(0.05,0.95,est_labelstr,transform=axs[r,c].transAxes,fontsize=fontsize,color='#40B5BC',verticalalignment='top')
        axs[r,c].text(0.05,0.95,corr_labelstr,transform=axs[r,c].transAxes,fontsize=fontsize,color='k',verticalalignment='top')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    os.makedirs('../outputs/', exist_ok=True)
    plt.savefig('../outputs/coco_grid_fig.pdf', dpi=10)

if __name__ == "__main__":
    with torch.no_grad():
        fix_randomness(seed=1043)
        # parsing args
        args = parser.parse_args()

        # dataset
        coco_val_2017_directory = '../data/val2017'
        coco_instances_val_2017_json = '../data/annotations_trainval2017/instances_val2017.json'
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

        rows = 2
        cols = 5 

        def scores_to_labels(x):
            tw = torch.where(x > args.th)
            est_labels = [[]]*x.shape[0]

            for k in tw[0].unique():
                est_labels[k] = [classes_list[idx] for idx in tw[1][tw[0]==0]]

            return est_labels

        imgs = []
        est_labels = []
        labels = []
        for i in range(rows*cols):
            img, est_label, label = random_example(dataset,model,scores_to_labels,corr,classes_list)
            imgs = imgs + [img]
            est_labels = est_labels + est_label
            labels = labels + label
        gridplot_imgs(imgs,est_labels,labels,rows,cols)
