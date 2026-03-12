import argparse
import copy
import datetime
import gc
import json
import math
import multiprocessing
import os
import pickle
import random
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2 as cv
import h5py
import numpy as np
import psutil
import scipy
import scipy.io as io
from PIL import Image
from referencing.typing import Anchor
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import metrics
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.switch_backend('agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms
from utils import *
from utils.util import (
    plotLossACC, saveFeatureMaps, saveGaborFilters,
    saveLossACC, saveParameters
)
from models import MyDataset
from models.ccnet import ccnet as co3net
from loss import (
    ArcFaceLoss, CenterLoss, SimplifiedSupConLoss,
    SupConLoss, UncertaintyWeighting
)
from MetaFuse import (
    SimpleWeightPredictor, test_with_weight_network, train_weight_network
)

#X-Anchor aggregation
def communication2(n, m, mode, server_model, models, client_weights):
    with torch.no_grad():
        if mode.lower() == 'fedmylove':

            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:

                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)


            for client_idx in range(len(client_weights)):
                for key in server_model.state_dict().keys():

                    if 'num_batches_tracked' in key:
                        continue


                    is_classification_layer = any(pattern in key for pattern in ['fc1', 'arclayer_', 'drop'])

                    if is_classification_layer:

                        print(f"Keeping personalized layer for client {client_idx}: {key}")
                        continue
                    else:

                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, models

#X-Anchor train
def fit_enhanced1(run_id, epoch, model, data_loader, optimize=None, phase='training', k=1, use_simplified=False,
                  global_model=None, current_com=1, total_com=4, client_mu=None, args=None,
                  arcface_criterion=None, center_criterion=None, paired_global_model=None,
                  cross_modal_global_model=None, uncertainty_weighting=None, ablation_config=None,
                  client_idx=None, weight_log_file=None):
    USE_CE = True
    USE_CONTRASTIVE = True
    USE_FEDPROX = True
    USE_ARCFACE = True
    USE_CENTER = True

    if phase not in ['training', 'testing']:
        raise TypeError('Invalid phase input!')

    if phase == 'training':
        model.train()
        if global_model is not None:
            global_model.train()
    if phase == 'testing':
        model.eval()
        if global_model is not None:
            global_model.eval()

    running_loss = 0
    entro_loss = 0
    supcon_loss = 0
    prox_loss_acc = 0
    arcface_loss_acc = 0
    center_loss_acc = 0
    running_correct = 0

    con_criterion = SimplifiedSupConLoss(temperature=args.temp).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    for batch_id, (datas, target) in enumerate(data_loader):
        data = datas[0].cuda()
        data_con = datas[1].cuda()
        data_con2 = datas[2].cuda()
        target = target.cuda()

        if phase == 'training':
            optimize.zero_grad()
            output, fe1 = model(data, target)
            _, fe2 = model(data_con, target)
            _, fe3 = model(data_con2, target)
            if global_model is not None:
                with torch.no_grad():
                    _, fe4 = global_model(data, target)
                    fe4 = fe4.detach()
            else:
                fe4 = None
            fe_list = [fe1, fe2, fe3]

        else:
            with torch.no_grad():
                output, fe1 = model(data, target)
                _, fe2 = model(data_con, target)
                _, fe3 = model(data_con2, target)
                if global_model is not None:
                    _, fe4 = global_model(data, target)
                    fe4 = fe4.detach()
                else:
                    fe4 = None
                fe_list = [fe1, fe2, fe3]

        if USE_CE:
            ce = criterion(output, target)
        else:
            ce = torch.tensor(0.0, device=output.device, requires_grad=True)
        if USE_CONTRASTIVE and len(fe_list) > 1:
            fe = torch.stack(fe_list, dim=1)
            ce2 = con_criterion(fe, target)
        else:
            ce2 = torch.tensor(0.0, device=output.device, requires_grad=True)

        mu_value = client_mu if client_mu is not None else args.mu
        if USE_FEDPROX and (global_model is not None) and (mu_value is not None) and (mu_value > 0):
            w_diff = torch.tensor(0.0, device=output.device)
            eps = 1e-12
            for w_global, w_local in zip(global_model.parameters(), model.parameters()):
                diff = w_local - w_global.detach()
                w_diff = w_diff + torch.sum(diff.pow(2))
            w_diff = torch.sqrt(w_diff + eps)
            loss2 = (mu_value / 2.0) * w_diff
        else:
            loss2 = torch.tensor(0.0, device=output.device, requires_grad=True)

        fe1_normalized = F.normalize(fe1, p=2, dim=1)

        if USE_ARCFACE and arcface_criterion is not None:
            ce4 = arcface_criterion(fe1_normalized, target)
        else:
            ce4 = torch.tensor(0.0, device=fe1.device, requires_grad=True)

        if USE_CENTER and center_criterion is not None:
            ce5 = center_criterion(fe1_normalized, target)
        else:
            ce5 = torch.tensor(0.0, device=fe1.device, requires_grad=True)

        if uncertainty_weighting is not None:
            active_losses = {}
            if USE_CE:
                active_losses['ce'] = ce
            if USE_CONTRASTIVE:
                active_losses['supcon'] = ce2
            if USE_FEDPROX:
                active_losses['fedprox'] = loss2
            if USE_ARCFACE:
                active_losses['arcface'] = ce4
            if USE_CENTER:
                active_losses['center'] = ce5

            if len(active_losses) > 0:
                loss, current_weights = uncertainty_weighting(active_losses)

                weight_idx = 0
                weight_ce = current_weights[weight_idx].item() if USE_CE else 0.0
                weight_idx += 1 if USE_CE else 0
                weight_supcon = current_weights[weight_idx].item() if USE_CONTRASTIVE else 0.0
                weight_idx += 1 if USE_CONTRASTIVE else 0
                weight_fedprox = current_weights[weight_idx].item() if USE_FEDPROX else 0.0
                weight_idx += 1 if USE_FEDPROX else 0
                weight_arcface = current_weights[weight_idx].item() if USE_ARCFACE else 0.0
                weight_idx += 1 if USE_ARCFACE else 0
                weight_center = current_weights[weight_idx].item() if USE_CENTER else 0.0
            else:
                loss = torch.tensor(0.0, device=output.device, requires_grad=True)
                weight_ce = weight_supcon = weight_fedprox = weight_arcface = weight_center = 0.0
        else:
            weight_ce = args.weight1 if USE_CE else 0.0
            weight_supcon = args.weight2 if USE_CONTRASTIVE else 0.0
            weight_fedprox = args.weight3 if USE_FEDPROX else 0.0
            weight_arcface = args.weight4 if USE_ARCFACE else 0.0
            weight_center = args.weight5 if USE_CENTER else 0.0
            loss = weight_ce * ce + weight_supcon * ce2 + weight_fedprox * loss2 + weight_arcface * ce4 + weight_center * ce5

        loss = loss.mean()

        running_loss += loss.data.cpu().numpy()
        entro_loss += ce.data.cpu().numpy()
        supcon_loss += ce2.data.cpu().numpy()
        prox_loss_acc += loss2.data.cpu().numpy()
        arcface_loss_acc += ce4.data.cpu().numpy() if isinstance(ce4, torch.Tensor) and ce4.requires_grad else 0.0
        center_loss_acc += ce5.data.cpu().numpy() if isinstance(ce5, torch.Tensor) and ce5.requires_grad else 0.0

        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().numpy()

        if phase == 'training':
            loss.backward()
            optimize.step()

    delta_theta = None

    total = len(data_loader.dataset)
    num_batches = len(data_loader)
    loss = running_loss / total
    entroloss = entro_loss * args.weight1 / total
    supconloss = supcon_loss * args.weight2 / total

    proxloss = prox_loss_acc / total
    arcfaceloss = arcface_loss_acc / total
    centerloss = center_loss_acc / total
    accuracy = (100.0 * running_correct) / total

    if epoch % 1 == 0 and phase == 'testing':
        if uncertainty_weighting is not None:
            current_weights = uncertainty_weighting.get_current_weights()
            log_vars = uncertainty_weighting.get_log_vars()

            active_loss_names = []
            if USE_CE:
                active_loss_names.append('CE')
            if USE_CONTRASTIVE:
                active_loss_names.append('SupCon')
            if USE_FEDPROX:
                active_loss_names.append('FedProx')
            if USE_ARCFACE:
                active_loss_names.append('ArcFace')
            if USE_CENTER:
                active_loss_names.append('Center')

            print(f'Epoch {epoch} [Com {current_com}/{total_com} :')
            print(f'  {phase} loss: {loss:.5f}, Accuracy: {running_correct}/{total} ({accuracy:.3f}%)')
            print(
                f'  原始损失值 - CE: {entroloss:.5f}, SupCon: {supconloss:.5f}, Prox: {proxloss:.5f}, ArcFace: {arcfaceloss:.5f}, Center: {centerloss:.5f}')

            if len(active_loss_names) > 0:
                weights_str = ', '.join(
                    [f'{name}: {current_weights[i]:.4f}' for i, name in enumerate(active_loss_names)])
                log_vars_str = ', '.join([f'{name}: {log_vars[i]:.4f}' for i, name in enumerate(active_loss_names)])
                print(f'  学习到的权重 (1/σ²) - {weights_str}')
                # print(f'  不确定性 log(σ²) - {log_vars_str}')
            if weight_log_file is not None and client_idx is not None:
                all_weights = uncertainty_weighting.get_current_weights()
                weight_ce = all_weights[0].item() if USE_CE else 0.0
                weight_supcon = all_weights[1].item() if USE_CONTRASTIVE else 0.0
                weight_fedprox = all_weights[2].item() if USE_FEDPROX else 0.0
                weight_arcface = all_weights[3].item() if USE_ARCFACE else 0.0
                weight_center = all_weights[4].item() if USE_CENTER else 0.0
                with open(weight_log_file, 'a') as f:
                    f.write(
                        f"{current_com}\t{epoch}\t{client_idx}\t{weight_ce:.6f}\t{weight_supcon:.6f}\t{weight_fedprox:.6f}\t{weight_arcface:.6f}\t{weight_center:.6f}\n")
        else:
            print(
                f'Epoch {epoch} [Com {current_com}/{total_com} 固定权重]: \t{phase} loss is \t{loss:.5f}, Entropy_loss is \t{entroloss:.5f}, SupCon_loss is \t{supconloss:.5f}, Prox_loss is \t{proxloss:.5f}, ArcFace_loss is \t{arcfaceloss:.5f}, Center_loss is \t{centerloss:.5f}; Accuracy is \t{running_correct}/{total} \t{accuracy:.3f}%')

    return loss, accuracy, delta_theta
