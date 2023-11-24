import os, argparse, time
import pickle
from pathlib import Path
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import numpy as np
import _init_paths
from pose_code.hourglass import hg
from pose_code.atthourglass import atthg
from pose_code import JointsMSELoss
from pose_code.utils import *
from train_utils import *
import copy
from tqdm import tqdm
from matplotlib import pyplot as plt
from visualizer import  Visualizer
import imageio as io


visualizer = Visualizer()


def get_optimizer(args, model):
    if args.solver == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.solver == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr)
    elif args.solver == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    else:
        print('Unknown solver: {}'.format(args.solver))
        assert False
    return optimizer


class Loss(torch.nn.Module):
    def __init__(self, device="cuda", stack_ri=0.8, ips=True, ips_lc=0.05):
        super().__init__()
        self.ips = ips
        self.ips_lc = ips_lc
        self.stack_ri = stack_ri
        self.sa_criterion = JointsMSELoss()
        
    def __get_joints_from_target(self, Y, X=None):
        b, c, h, w = Y.shape
        batch_joints = []
        for b in range(Y.shape[0]):
            joints = []
            for c in range(Y.shape[1]):
                gt = Y.detach().cpu().numpy()
                y = np.uint8(np.where(gt[b,c]>0, 255, 0))
                if y.sum() > 0:
                    num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(y)
                    center = [centers[1][1]/h, centers[1][0]/w] #0 for background
                else:
                    center = [0.,0.]
                joints.append(center)
            batch_joints.append(joints)
        batch_joints = torch.tensor(np.array(batch_joints)).to(Y.device)
        return batch_joints

    # def __get_joints_from_target(self, Y, X=None):
    #     b, c, h, w = Y.shape
    #     indices = torch.argmax(Y.view(b, c, -1), dim=2)
    #     i_indices = indices // w
    #     j_indices = indices %  w
    #     i_indices = i_indices.view(b, c, 1) / h
    #     j_indices = j_indices.view(b, c, 1) / w
    #     batch_joints = torch.cat((i_indices, j_indices), dim=2)
    #     return batch_joints
    
    def _calc_wpdd(self, pd, gt, vis, alpha=0.8):
        wpdds = 0
        b_num, j_num, dim = pd.shape 
        for b in range(b_num):
            wpdd = 0
            for j1 in range(j_num-1):
                for j2 in range(j1+1, j_num):
                    if vis[b, j1] and vis[b, j2]:
                        gt_dist = torch.norm(gt[b,j1]-gt[b,j2], 2)
                        pd_dist = torch.norm(pd[b,j1]-pd[b,j2], 2)
                        wpdd += (gt_dist-pd_dist)**2 * alpha**(j2-j1)
            wpdds += wpdd/b_num
        return wpdds

    def _calc_sk_loss(self, batch_joints, sgi, vis):
        satt, mean_js, var_js, vis_js = sgi
        # calculate absolute disk position distances 
        js_diff = torch.abs(batch_joints-mean_js)
        joints_pos_loss_mean = torch.mean(torch.pow(js_diff, 2)*vis)
        # calculate loss for distance between two continues disks between gt and sk
        joints_dis_loss_mean = self._calc_wpdd(mean_js, batch_joints, vis, alpha=0.8)
        
        joints_loss_var = torch.abs(torch.mean(var_js) - 1)
        loss = 0.75*joints_pos_loss_mean + 0.25*joints_dis_loss_mean
        return loss, vis_js

    def forward(self, output, target, sgi, vis, X):
        sk_loss, sa_loss = 0, 0
        gt_batch_joints = self.__get_joints_from_target(target, X)

        sum_of_coeffs = 0
        if isinstance(output, list) and isinstance(sgi, list):
            for i, (o, s) in enumerate(zip(output[::-1], sgi[::-1])):
                coeff = self.stack_ri**i
                sum_of_coeffs += coeff

                sa_loss += self.sa_criterion(o, target, vis) * coeff
                g_l, sk_vis = self._calc_sk_loss(gt_batch_joints, s, vis)
                sk_loss += g_l * coeff
                
            sa_loss = sa_loss/sum_of_coeffs
            sk_loss = sk_loss/sum_of_coeffs
        else: # single output
            sk_loss, sk_vis = self._calc_sk_loss(gt_batch_joints, sgi, vis)
            sa_loss = self.sa_criterion(output, target, vis)
        
        sa_loss *= 5e4
        sk_loss *= 1e1
        if self.ips:
            loss = (1-self.ips_lc)*sa_loss + self.ips_lc*sk_loss
        else:
            loss = sa_loss
            
        self.visualize_training_process(X, target, output, sgi, vis, prob=0.02)
        
        loss_infos = {
            "sa": sa_loss,
            "sk": sk_loss,
        }
        return loss, loss_infos
    
    def visualize_training_process(self, X, target, output, sgi, vis, prob=0.02):
        if torch.rand(1).item() > (1-prob):
            visualizer.load(X[0], target, [], output, sgi, torch.zeros_like(X[0]))
            att = visualizer.get_hca_atts()
            joints = sgi[-1][1][0].detach().cpu().numpy()
            im = X[0].detach().cpu().numpy().transpose([1,2,0])
            ov = visualizer.get_overlayed_vertebral(joints, im, vis[0])
            io.imwrite("vis/att.png", att)
            io.imwrite("vis/sk.png", ov)


def train(args, model, tr_loader, vl_loader, index_of_joints, device, writer):
    best_acc = 0

    # define loss function (criterion) and optimizer
    criterion = Loss(ips=args.ips, stack_ri=args.stack_ri, ips_lc=args.ips_lc).to(device)
    optimizer = get_optimizer(args, model)    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=11, factor=0.5)

    # optionally resume from a checkpoint
    if args.resume:
        print("=> loading checkpoint to continue learing process")
        model.load_state_dict(torch.load(f'{args.weigths_dir}/{args.name}/{args.modality}_{args.stacks}', map_location='cpu')['model_weights'])

    # train and eval
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        # lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print(f'\nEpoch: {epoch+1:03d}/{args.epochs} | LR: {optimizer.param_groups[0]["lr"]}')

        # decay sigma
        if args.sigma_decay > 0:
            tr_loader.dataset.sigma *= args.sigma_decay
            vl_loader.dataset.sigma *= args.sigma_decay

        # train for one epoch
        train_loss, train_acc = train_one_epoch(tr_loader, model, criterion, optimizer, index_of_joints, device, writer)
        scheduler.step(train_loss)
        
        # evaluate on validation set
        valid_loss, valid_acc = validate_one_epoch(vl_loader, model, criterion, index_of_joints, device, writer)

        writer.add_scalars(f"loss/tr-vl - {args.name}", {"tr": train_loss, "vl": valid_loss}, epoch)
        writer.add_scalars(f"acc/tr-vl - {args.name}", {"tr": train_acc, "vl": valid_acc}, epoch)
        
        # remember best acc and save checkpoint
        if valid_acc > best_acc:
           state = copy.deepcopy({'model_weights': model.state_dict()})
           torch.save(state, f'{args.weigths_dir}/{args.name}/{args.modality}_{args.stacks}')
           best_acc = valid_acc


def train_one_epoch(train_loader, model, criterion, optimizer, index_of_joints, device, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    loss_dices = AverageMeter()

    # switch to train mode
    model.train()

    loop = tqdm(train_loader, desc="Training")
    itr = 0
    for (input, target, vis) in loop:
        itr += 1
        input, target = input.to(device), target.to(device, non_blocking=True)
        vis = vis.to(device, non_blocking=True)
        # compute output
        output, sgi = model(input)
       
        loss, loss_infos = criterion(output, target, sgi, vis, input)
        step = ep*len(train_loader)+itr
        if ep>0:
            writer.add_scalars("loss-tr", {"loss": loss, "sa": loss_infos["sa"], "sk": loss_infos["sk"]}, step)
        if isinstance(output, list): 
            output = output[-1]
            sgi = sgi[-1]
        acc = accuracy(output, target, index_of_joints)
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        acces.update(acc[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f'iter: {step:05d} | avg-loss: {losses.avg:.7f} | loss_sa: {loss_infos["sa"].item():0.7f} | loss_sk: {loss_infos["sk"].item():0.7f} | Acc: {acces.avg:.7f}')
    return losses.avg, acces.avg

ep = 0
def validate_one_epoch(val_loader, model, criterion, index_of_joints, device, writer):
    global ep
    Flag_visualize = True
    ep += 1
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    loss_dices = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    loop = tqdm(val_loader, desc="Validation")
    itr = 0
    with torch.no_grad():
        for (input, target, vis) in loop:
            itr += 1
            # save_att_output = get_att_layer_output(model)
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            vis = vis.to(device, non_blocking=True)
            # compute output
            output, sgi = model(input) 
            loss, loss_infos = criterion(output, target, sgi, vis, input)
        
            if isinstance(output, list): 
                output = output[-1]
                sgi = sgi[-1]
            acc = accuracy(output.cpu(), target.cpu(), index_of_joints)
            loss_dice = dice_loss(output, target)
            # if args.visualize:
            #     save_attention(
            #         input[0:1], output[0:1], target[0:1], 
            #         torch.sum(save_att_output.outputs[0][0], dim=0), 
            #         target_th=0.5, save_dir=args.vis_dir, id=i)
    
            # if Flag_visualize and args.visualize:
            #     # save the visualization only for the first batch of the validation
            #     # save_epoch_res_as_image2(input, output, target, epoch_num=ep, target_th=0.6, save_dir="./visualize")
            #     ind = -3
            #     save_attention(
            #         input[ind:], output[ind:], target[ind:], 
            #         torch.sum(save_att_output.outputs[0][ind], dim=0), 
            #         target_th=0.6, save_dir="./visualize", id=ep)
            #     Flag_visualize = False
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            acces.update(acc[0], input.size(0))
            loss_dices.update(loss_dice.item(), input.size(0))
            loop.set_description(f'iter: {itr:02d}/{len(val_loader)} | avg-loss: {losses.avg:.7f} | loss_sa: {loss_infos["sa"].item():0.7f}| loss_sk: {loss_infos["sk"]:0.7f} | Acc: {acces.avg:.7f} | Dice: {loss_dices.avg:.5f}')

    return losses.avg, acces.avg
