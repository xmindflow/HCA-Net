import os
import argparse
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import numpy as np
import _init_paths
from pathlib import Path
import pickle
from models.model import HCANet
from pose_code import JointsMSELoss
from pose_code.utils import *
from train_utils import *
from torch.utils.data import DataLoader 
import copy
import random 
from tqdm import tqdm
from train import train
from test import test
from torch.utils.tensorboard import SummaryWriter


idx = []

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(args):
    print(f"==> creating model {args.name}, stacks={args.stacks}, blocks={args.blocks}")
    model = HCANet(
        num_stacks=args.stacks, num_blocks=args.blocks, 
        num_classes=args.njoints,
        ips=args.ips,
        ips_feedback=args.ips_feedback,
        ips_feedback_sum=args.ips_feedback_sum)
    model = torch.nn.DataParallel(model).to(device)
    return model


def main(args):
    writer = SummaryWriter(log_dir=f"./runs/{args.name}-{args.modality}")
    Path(f"{args.weigths_dir}/{args.name}").mkdir(exist_ok=True, parents=True)
    Path(f"vis").mkdir(exist_ok=True, parents=True)
    
    if args.mode == "train":
        data_path = f'{args.datapath}/prepared_trainset_{args.modality}'
    elif args.mode == "test":
        data_path = f'{args.datapath}/prepared_testset_{args.modality}_full'

    # Load the prepared dataset
    with open(data_path, 'rb') as file_pi:
        full = pickle.load(file_pi)
    full[0] = full[0][:, :, :, :, 0]

    model = get_model(args)

    ## Get the visualization resutls of the test set
    if args.mode == 'train':
        tr_idx = int(np.round(len(full[0])*0.9))
        full_dataset_tr = image_Dataset(image_paths=full[0][:tr_idx], target_paths=full[1][:tr_idx], use_flip=True, target_th=args.gt_th)
        full_dataset_vl = image_Dataset(image_paths=full[0][tr_idx:], target_paths=full[1][tr_idx:], use_flip=False, target_th=args.gt_th)
        MRI_tr_loader = DataLoader(full_dataset_tr, batch_size=args.train_batch, shuffle=True, num_workers=0)
        MRI_vl_loader = DataLoader(full_dataset_vl, batch_size=args.val_batch, shuffle=False, num_workers=0)
    else: # test
        full_dataset_te = image_Dataset(image_paths=full[0], target_paths=full[1], use_flip=False, target_th=args.gt_th)
        MRI_te_loader = DataLoader(full_dataset_te, batch_size= 1, shuffle=False, num_workers=0)

    # idx is the index of joints used to compute accuracy (we detect 11 joints starting from C1 to ...) 
    index_of_joints = range(1, args.njoints+1)

    if args.mode == 'train':
        train(
            args=args,
            model=model,
            tr_loader=MRI_tr_loader,
            vl_loader=MRI_vl_loader,
            index_of_joints=index_of_joints,
            device=device,
            writer=writer,
        )

    else: # test
        Path(f"vis/{args.name}/out_{args.modality}").mkdir(exist_ok=True, parents=True)

        model.load_state_dict(torch.load(f'{args.weigths_dir}/{args.name}/{args.modality}_{args.stacks}', map_location='cpu')['model_weights'])
        test(
            args=args,
            model=model,
            full = full,
            te_loader=MRI_te_loader,
            index_of_joints=index_of_joints,
            device=device,
            writer=writer
        )


def show_attention(val_loader, model):
    ## define the attention layer output
    save_att_output = get_att_layer_output(model)

    # switch to evaluate mode
    N = 1
    Sel= 0
    model.eval()
    with torch.no_grad():
        for i, (input, target, vis) in enumerate(val_loader):
            if i==Sel:
                input  = input [N:N+1]
                target = target[N:N+1]
                input  = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(input)
                att = save_att_output.outputs[0][0,0]
                output = output[-1]
                save_attention(input, output, target, att, target_th=0.6)
    return 0, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verterbal disc labeling using pose estimation')
    
    ## Parameters
    parser.add_argument('--datapath', default='/hdd/datasets/ivd_dataset', type=str, help='Dataset address')
    parser.add_argument('--mode', choices=['train', 'test'], type=str, help='Running mode: train, test')
    parser.add_argument('--name', default="HCA-Net", type=str, help='Model name')
    parser.add_argument('--weigths-dir', default='./weights', type=str, help='weigths directory')
    parser.add_argument('--njoints', default=11, type=int, help='Number of joints')
    parser.add_argument('--resume', action="store_true", help='Resume the training from the last checkpoint')  
    parser.add_argument('--attshow', default= False, type=bool, help='Show the attention map') 
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--modality', default='t1', type=str, metavar='N', help='Data modality')
    parser.add_argument('--train-batch', default=3, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--val-batch', default=4, type=int, metavar='N', help='validation batchsize')
    parser.add_argument('--solver', metavar='SOLVER', default='rms', choices=['rms', 'adam', 'sgd'], help='optimizers')
    parser.add_argument('--lr', '--learning-rate', default=0.00025, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90], help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('-e', '--evaluate', default=False, type=bool, help='evaluate model on validation set')
    parser.add_argument('--att', default=False, type=bool, help='Use attention or not')
    parser.add_argument('-s', '--stacks', default=2, type=int, metavar='N', help='Number of hourglasses to stack')
    parser.add_argument('--features', default=256, type=int, metavar='N', help='Number of features in the hourglass')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N', help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--sigma-decay', type=float, default=0, help='Sigma decay rate for each epoch.')
    parser.add_argument('--visualize', action="store_true", help='Save visualizations')
    parser.add_argument('--vis-dir', default="res", type=str, help='The directory to save visualization images')
    parser.add_argument('--ips', action='store_false', help="use IPS module")
    parser.add_argument('--ips-feedback', action='store_false', help="feedback to the HCA module using the IPS attention")
    parser.add_argument('--ips-feedback-sum', action='store_false', help="use summation instead of multiplication at IPS feedback")
    parser.add_argument('--stack-ri', type=float, default=1.0, help='Stack loss ratio (0-1)')
    parser.add_argument('--ips-lc', type=float, default=0.5, help='ips loss coef.')
    parser.add_argument('--gt-th', type=float, default=0.1, help='gt target threshold binary')

    main(parser.parse_args())
