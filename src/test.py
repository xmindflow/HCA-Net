## Original code from Neuropoly, Lucas
## Code modified by Reza Azad
from __future__ import print_function, absolute_import
from Metrics import *
import torch
import numpy as np
from train_utils import *
import pickle
from pathlib import Path
import cv2
from sklearn.utils.extmath import cartesian
from visualizer import Visualizer
from PIL import Image
import imageio


# intialize metrics
distance_l2 = []
zdis = []
faux_pos = []
faux_neg = []
tot = []


def prediction_coordinates(final, coord_gt):
    num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(np.uint8(np.where(final>0, 255, 0)))
    #centers = peak_local_max(final, min_distance=5, threshold_rel=0.3)

    centers = centers[1:] #0 for background
    coordinates = []
    for x in centers:
        coordinates.append([x[0], x[1]])
    #print('calculating metrics on image')
    mesure_err_disc(coord_gt, coordinates, distance_l2)
    mesure_err_z(coord_gt, coordinates, zdis)
    fp = Faux_pos(coord_gt, coordinates, tot)
    fn = Faux_neg(coord_gt, coordinates)
    faux_pos.append(fp)
    faux_neg.append(fn)
    
    return coordinates
   
    
## Functions from neuropoly
def retrieves_gt_coord(ds):
    coord_retrieved = []
    for i in range(len(ds[1])):
        coord_tmp = [[], []]
        for j in range(len(ds[1][i])):
            if ds[1][i][j][3] == 1 or ds[1][i][j][3] > 30:
                print('remove' + str(ds[1][i][j][3]))
                pass
            else:
                coord_tmp[0].append(ds[1][i][j][2])
                coord_tmp[1].append(ds[1][i][j][1])
        coord_retrieved.append(coord_tmp)
    return (coord_retrieved)
 

def test(args, model, full, te_loader, index_of_joints, device, writer):

    with open(f'{args.datapath}/prepared_testset_{args.modality}_ds',   'rb') as file_pi:       
        ds = pickle.load(file_pi)
        coord_gt = retrieves_gt_coord(ds)
    print('retrieving ground truth coordinates')
    norm_mean_skeleton = np.load(f'{args.datapath}/{args.modality}_Skelet.npy')

    visualizer = Visualizer()
    model.eval()
    for i, (input, target, vis) in enumerate(te_loader):
        input, target = input.to(device), target.to(device, non_blocking=True)
        
        assert input.shape[0] == 1, "Batch size for the test data loader must be 1!"
        outs, ipses = model(input)
        
        # save last output block results
        visualizer.load(input[0], target[0], coord_gt[i], outs, ipses, None)
        out_atts = visualizer.get_hca_atts()
        imageio.imwrite(f"vis/{args.name}/out_{args.modality}/{i}.jpg", out_atts)
        
        output = outs[-1]        
        x = full[0][i]
        prediction = extract_skeleton(input, output, target, norm_mean_skeleton, flag_save=False)
        
        if args.visualize:
            Path(f"vis/{args.name}/{args.vis_dir}_{args.modality}").mkdir(parents=True, exist_ok=True)
            save_attention(
                input[0:1], output[0:1], target[0:1], 
                output[0].sum(axis=0), 
                target_th=0.8, save_dir=f"vis/{args.name}/{args.vis_dir}_{args.modality}", id=i)

        prediction = np.sum(prediction[0], axis = 0)
        prediction = np.rot90(prediction,3)
        prediction = cv2.resize(prediction, (x.shape[0], x.shape[1]), interpolation=cv2.INTER_NEAREST)
        coordinates = prediction_coordinates(prediction, coord_gt[i])


    print('distance med l2 and std ' + str(np.median(distance_l2)))
    print(np.std(distance_l2))
    print('distance med z and std ' + str(np.mean(zdis)))
    print(np.std(zdis))
    print('faux neg per image ', faux_neg)
    print('total number of points ' + str(np.sum(tot)))
    print('number of faux neg ' + str(np.sum(faux_neg)))
    print('number of faux pos ' + str(np.sum(faux_pos)))
    print('False negative percentage ' + str(np.sum(faux_neg)/ np.sum(tot)*100))
    print('False positive percentage ' + str(np.sum(faux_pos)/ np.sum(tot)*100))
    

def check_skeleton(cnd_sk, mean_skeleton):
    cnd_sk = np.array(cnd_sk)
    Normjoint = np.linalg.norm(cnd_sk[0]-cnd_sk[4])
    for idx in range(1, len(cnd_sk)):
        cnd_sk[idx] = (cnd_sk[idx] - cnd_sk[0]) / Normjoint
    cnd_sk[0] -= cnd_sk[0]
    return np.sum(np.linalg.norm(mean_skeleton[:len(cnd_sk)]-cnd_sk))
    
  
idtest = 1
from scipy.ndimage import gaussian_filter

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def extract_skeleton(inputs, outputs, target, norm_mean_skeleton, flag_save=False, target_th=0.5):
    global idtest
    # outputs = torch.sigmoid(outputs)
    outputs = outputs.data.cpu().numpy()
    target = target.data.cpu().numpy()
    inputs = inputs.data.cpu().numpy()
    skeleton_images = []
    
    b, c, h, w = outputs.shape
    for idx in range(b):    
        count_list = []
        Nch = 0
        center_list = {}

        while np.sum(np.sum(target[idx, Nch]))>0:
            Nch += 1
        Final = np.zeros((b, Nch, h, w))      
        for idy in range(Nch): 
            ych = outputs[idx, idy]
            ych = np.rot90(ych)
            # ych = sigmoid(ych)
            # ych = gaussian_filter(ych, 0.2)         
            ych = normalize(ych)
            ych[np.where(ych<target_th)] = 0
            Final[idx, idy] = ych
            ych = np.where(ych>0, 1.0, 0)
            ych = np.uint8(ych)
            num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(ych)
            
            if num_labels!=2:
                mean_centers = np.mean(centers[1:], 0).tolist()
                count_list.append(num_labels)
                center_list[str(idy)] = [t[::-1] for t in centers[1:].tolist()+[mean_centers]]
            else:
                count_list.append(num_labels-1)
                center_list[str(idy)] = [t[::-1] for t in centers[1:].tolist()]
        
        ups = []
        for c in count_list:
            ups.append(range(c))
        combs = cartesian(ups)
        best_loss = np.Inf
        best_skeleton = []
        for comb in combs:
            cnd_skeleton = []
            for joint_idx, cnd_joint_idx in enumerate(comb):
                cnd_center = center_list[str(joint_idx)][cnd_joint_idx]
                cnd_skeleton.append(cnd_center)
            loss = check_skeleton(cnd_skeleton, norm_mean_skeleton)
            if best_loss > loss:
                best_loss = loss
                best_skeleton = cnd_skeleton
        Final2  = np.uint8(np.where(Final>0, 1, 0))
        cordimg = np.zeros(Final2.shape)
        hits = np.zeros_like(outputs[0])
        for i, jp, in enumerate(best_skeleton):
            jp = [int(t) for t in jp]
            hits[i, jp[0]-1:jp[0]+2, jp[1]-1:jp[1]+2] = [255, 255, 255]
            hits[i, :, :] = cv2.GaussianBlur(hits[i, :, :],(5,5),cv2.BORDER_DEFAULT)
            hits[i, :, :] = hits[i, :, :]/hits[i, :, :].max()*255
            cordimg[idx, i, jp[0], jp[1]] = 1
        
        for id_ in range(Final2.shape[1]):
            num_labels, labels_im = cv2.connectedComponents(Final2[idx, id_])
            for id_r in range(1, num_labels):
                if np.sum(np.sum((labels_im==id_r) * cordimg[idx, id_]))>0:
                   labels_im = labels_im == id_r
                   continue
            Final2[idx, id_] = labels_im
        Final = Final * Final2
        
        skeleton_images.append(hits)
        
    skeleton_images = np.array(skeleton_images)
    inputs = np.rot90(inputs, axes=(-2, -1))
    target = np.rot90(target, axes=(-2, -1))
    if flag_save:
      save_test_results(inputs, skeleton_images, targets=target, name=idtest, target_th=0.5)
    idtest+=1
    return Final
    
 
def save_test_results(inputs, outputs, targets, name='', target_th=0.5):
    clr_vis_Y = []
    hues = np.linspace(0, 179, targets.shape[1], dtype=np.uint8)
    blank_ch = 255*np.ones_like(targets[0,0], dtype=np.uint8)

    for Y in [targets, outputs]:
        for y, x in zip(Y, inputs):
            y_colored = np.zeros([y.shape[1], y.shape[2], 3], dtype=np.uint8)
            y_all = np.zeros([y.shape[1], y.shape[2]], dtype=np.uint8)
            
            for ych, hue_i in zip(y, hues):
                ych = ych/np.max(np.max(ych))
                ych[np.where(ych<target_th)] = 0
                ych = cv2.GaussianBlur(ych,(15,15),cv2.BORDER_DEFAULT)

                ych_hue = np.ones_like(ych, dtype=np.uint8)*hue_i
                ych = np.uint8(255*ych/np.max(ych))

                colored_ych = np.zeros_like(y_colored, dtype=np.uint8)
                colored_ych[:, :, 0] = ych_hue
                colored_ych[:, :, 1] = blank_ch
                colored_ych[:, :, 2] = ych
                colored_y = cv2.cvtColor(colored_ych, cv2.COLOR_HSV2BGR)

                y_colored += colored_y
                y_all += ych

            x = np.moveaxis(x, 0, -1)
            x = x/np.max(x)*255

            x_3ch = np.zeros([x.shape[0], x.shape[1], 3])
            for i in range(3):
                x_3ch[:, :, i] = x[:, :, 0]

            img_mix = np.uint8(x_3ch*0.5 + y_colored*0.5)
            # img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
            clr_vis_Y.append(img_mix)

    
    t = np.array(clr_vis_Y)
    t = np.transpose(t, [0, 3, 1, 2])
    trgts = make_grid(torch.Tensor(t), nrow=4)

    txt = f'./visualize/{name}_test_result.png'
    res = np.transpose(trgts.numpy(), (1,2,0))
    cv2.imwrite(txt, res)
