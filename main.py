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
from XAnchor import communication2, fit_enhanced1
parser = argparse.ArgumentParser(
    description="Fed-XCal"
)

parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--epoch_num", type=int, default=3)
parser.add_argument("--com", type=int, default=100)

parser.add_argument("--temp", type=float, default=0.07)
parser.add_argument("--weight1", type=float, default=0.7)
parser.add_argument("--weight2", type=float, default=0.15)
parser.add_argument("--weight3", type=float, default=1.0)
parser.add_argument("--weight4", type=float, default=0.01)
parser.add_argument("--weight5", type=float, default=0.1)
parser.add_argument("--mu", type=float, default=1e-2)
parser.add_argument("--arcface_s", type=float, default=64.0)
parser.add_argument("--arcface_m", type=float, default=0.5)
parser.add_argument("--center_alpha", type=float, default=0.5)
parser.add_argument("--feat_dim", type=int, default=6144)
parser.add_argument("--lr", type=float, default=0.001)

parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--wn_lr", type=float, default=0.01)
parser.add_argument("--wn_weight_decay", type=float, default=1e-4)
parser.add_argument("--wn_step_size", type=int, default=50)
parser.add_argument("--wn_gamma", type=float, default=0.8)
parser.add_argument("--wn_epochs", type=int, default=500)

parser.add_argument("--id_num", type=int, default=1000)
parser.add_argument("--gpu_id", type=str, default='1')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_save_dir", type=str, default="./Results/Saved models/")
args = parser.parse_args()

import warnings

warnings.simplefilter('ignore', category=FutureWarning)

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# Testing for single-modal
def save_roc_data_and_plot(scores, labels, save_dir, test_name):

    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        import json


        os.makedirs(save_dir, exist_ok=True)


        binary_labels = np.array([(1 if label == 1 else 0) for label in labels])


        similarity_scores = 1 - np.array(scores)


        fpr, tpr, thresholds = roc_curve(binary_labels, similarity_scores)
        roc_auc = auc(fpr, tpr)


        roc_data = {
            'scores': scores.tolist() if isinstance(scores, np.ndarray) else list(scores),
            'labels': labels.tolist() if isinstance(labels, np.ndarray) else list(labels),
            'binary_labels': binary_labels.tolist(),
            'similarity_scores': similarity_scores.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': roc_auc,
            'test_name': test_name
        }


        json_file = os.path.join(save_dir, 'roc_data.json')
        with open(json_file, 'w') as f:
            json.dump(roc_data, f, indent=2)


        txt_file = os.path.join(save_dir, 'roc_data.txt')
        with open(txt_file, 'w') as f:
            f.write(f"# ROC Data for {test_name}\n")
            f.write(f"# AUC: {roc_auc:.6f}\n")
            f.write("# Format: FPR TPR Threshold\n")
            for i in range(len(fpr)):
                f.write(f"{fpr[i]:.6f} {tpr[i]:.6f} {thresholds[i]:.6f}\n")


        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {test_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)


        img_file = os.path.join(save_dir, 'roc_curve.png')
        plt.savefig(img_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"ROC数据已保存到: {save_dir}")
        print(f"  - JSON文件: {json_file}")
        print(f"  - TXT文件: {txt_file}")
        print(f"  - ROC图片: {img_file}")
        print(f"  - AUC: {roc_auc:.6f}")

        return roc_auc

    except Exception as e:
        print(f"保存ROC数据时出错: {e}")
        return None
def test0(model, gallery_file, query_file,  roc_save_dir=None, test_name="test0"):
    global avgeer, avgacc
    print('Start Testing!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    trainset = MyDataset(txt=gallery_file, transforms=None, train=True)
    testset = MyDataset(txt=query_file, transforms=None, train=False)

    batch_size = 512  # 128

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=2)
    data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=2)

    fileDB_train = getFileNames(gallery_file)
    fileDB_test = getFileNames(query_file)

    net = model

    net.cuda()
    net.eval()

    featDB_train = []
    iddb_train = []

    for batch_id, (datas, target) in enumerate(data_loader_train):
        data = datas[0]

        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode(data)
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_train = codes
            iddb_train = y
        else:
            featDB_train = np.concatenate((featDB_train, codes), axis=0)
            iddb_train = np.concatenate((iddb_train, y))

    print('completed feature extraction for training set.')
    print('featDB_train.shape: ', featDB_train.shape)

    classNumel = len(set(iddb_train))
    num_training_samples = featDB_train.shape[0]
    trainNum = num_training_samples // classNumel
    print('[classNumel, imgs/class]: ', classNumel, trainNum)
    print('\n')

    featDB_test = []
    iddb_test = []

    print('Start Test Feature Extraction.')
    for batch_id, (datas, target) in enumerate(data_loader_test):

        data = datas[0]
        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode(data)

        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            iddb_test = y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))

    s = []  # matching score
    l = []  # intra-class or inter-class matching
    ntest = featDB_test.shape[0]
    ntrain = featDB_train.shape[0]

    for i in range(ntest):
        feat1 = featDB_test[i]

        for j in range(ntrain):
            feat2 = featDB_train[j]

            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi

            s.append(dis)

            if iddb_test[i] == iddb_train[j]:  # same palm
                l.append(1)
            else:
                l.append(-1)

    # 保存ROC数据和绘制ROC曲线
    if roc_save_dir is not None:
        save_roc_data_and_plot(s, l, roc_save_dir, test_name)

    print('\n------------------')
    print('Rank-1 acc of the test set...')

    def calculate_rank1_accuracy(iddb_test, iddb_train, fileDB_test, fileDB_train, s, ntest, ntrain):
        global avgacc
        cnt = 0
        corr = 0
        for i in range(ntest):
            probeID = iddb_test[i]

            dis = np.zeros((ntrain, 1))

            for j in range(ntrain):
                dis[j] = s[cnt]
                cnt += 1

            idx = np.argmin(dis[:])

            galleryID = iddb_train[idx]

            if probeID == galleryID:
                corr += 1

        rankacc = corr / ntest * 100
        avgacc = avgacc + rankacc
        print('rank-1 acc: %.3f%%' % rankacc)
        print('-----------')

    calculate_rank1_accuracy(iddb_test, iddb_train, fileDB_test, fileDB_train, s, ntest, ntrain)


# Train dataset
def Dataset2():

    from models.dataset import MyDataset2

    #BLUE ↔ NIR
    src_dataset_1 = DataLoader(
        MyDataset2(
            txt='./Data//train_MSBLUE .txt',
            txt_auxiliary='./Data/train_MSNIR .txt',
            transforms=None, train=True, imside=128, outchannels=1
        ), batch_size=batch_size, num_workers=2, shuffle=True)

    src_dataset_2 = DataLoader(
        MyDataset2(
            txt='./Data/train_MSNIR .txt',
            txt_auxiliary='./Data/train_MSBLUE .txt',
            transforms=None, train=True, imside=128, outchannels=1
        ), batch_size=batch_size, num_workers=2, shuffle=True)

    #WHT ↔ 700
    src_dataset_3 = DataLoader(
        MyDataset2(
            txt='./Data/train_WHT.txt',
            txt_auxiliary='./Data/train_700.txt',
            transforms=None, train=True, imside=128, outchannels=1
        ), batch_size=batch_size, num_workers=2, shuffle=True)

    src_dataset_4 = DataLoader(
        MyDataset2(
            txt='./Data/train_700.txt',
            txt_auxiliary='./Data/train_WHT.txt',
            transforms=None, train=True, imside=128, outchannels=1
        ), batch_size=batch_size, num_workers=2, shuffle=True)

    #Tongji_print ↔ Tongji_vein
    src_dataset_5 = DataLoader(
        MyDataset2(
            txt='./Data/Tongji_print_train.txt',
            txt_auxiliary='./Data/Tongji_vein_train.txt',
            transforms=None, train=True, imside=128, outchannels=1
        ), batch_size=batch_size, num_workers=2, shuffle=True)

    src_dataset_6 = DataLoader(
        MyDataset2(
            txt='./Data/Tongji_vein_train.txt',
            txt_auxiliary='./Data/Tongji_print_train.txt',
            transforms=None, train=True, imside=128, outchannels=1
        ), batch_size=batch_size, num_workers=2, shuffle=True)

    dataloaders = []
    dataloaders.append(src_dataset_1)
    dataloaders.append(src_dataset_2)
    dataloaders.append(src_dataset_3)
    dataloaders.append(src_dataset_4)
    dataloaders.append(src_dataset_5)
    dataloaders.append(src_dataset_6)
    return dataloaders


if __name__ == "__main__":

    for run_id in range(1):

        warnings.simplefilter('ignore', category=FutureWarning)

        set_seed(args.seed)
        batch_size = args.batch_size
        epoch_num = args.epoch_num
        num_classes = args.id_num
        weight1 = args.weight1
        weight2 = args.weight2
        weight3 = args.weight3
        weight4 = args.weight4
        weight5 = args.weight5
        model_save_dir = args.model_save_dir
        mu = args.mu
        communications = args.com
        feat_dim = args.feat_dim

        avgeer = 0
        avgacc = 0
        client_weights = [1 / 3 for i in range(3)]



        print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

        # Datasets
        train_set_files = [
            './Data/train_MSBLUE .txt',
            './Data/train_MSNIR .txt',
            './Data/train_WHT.txt',
            './Data/train_700.txt',
            './Data/Tongji_print_train.txt',
            './Data/Tongji_vein_train.txt',
        ]
        test_set_files = [
            './Data/test_MSBLUE.txt',
            './Data/test_MSNIR.txt',
            './Data/test_WHT.txt',
            './Data/test_700.txt',
            './Data/Tongji_print_test.txt',
            './Data/Tongji_vein_test.txt',
        ]
        names = ['BLUE', 'NIR', 'WHT', '700', 'Tongji_print', 'Tongji_vein']

        train_datas = Dataset2()

        #Models
        server_model = co3net(num_classes=num_classes).cuda()
        prox_server_model1 = copy.deepcopy(server_model)
        prox_server_model2 = copy.deepcopy(server_model)
        weight_predictor = SimpleWeightPredictor(input_dim=num_classes, hidden_dim=args.hidden_dim)
        weight_predictors = [copy.deepcopy(weight_predictor) for idx in range(3)]
        models = [copy.deepcopy(server_model) for idx in range(6)]



        #Loss
        criterion = nn.CrossEntropyLoss()
        arcface_criterion = ArcFaceLoss(num_classes=args.id_num, feat_dim=feat_dim, s=args.arcface_s, m=args.arcface_m).cuda()
        center_criterion = CenterLoss(num_classes=args.id_num, feat_dim=feat_dim, alpha=args.center_alpha).cuda()
        uncertainty_weighting = UncertaintyWeighting(num_tasks=5).cuda()

        # Optimizers
        optimizers = [torch.optim.Adam(models[idx].parameters(), lr=args.lr) for idx in range(6)]

        for idx in range(6):
            optimizers[idx] = torch.optim.Adam([
                {'params': models[idx].parameters(), 'lr': args.lr},
                {'params': uncertainty_weighting.parameters(), 'lr': args.lr}
            ])

        #X-Anchor
        for com in range(args.com):
            for idx in range(6):
                if idx in [0, 2, 4]:
                    current_global_model = prox_server_model1
                else:
                    current_global_model = prox_server_model2
                for epoch in range(args.epoch_num):
                    epoch_info = f"[Epoch {epoch + 1}/{args.epoch_num}]"

                    train_loss, train_accuracy, _ = fit_enhanced1(
                        run_id, epoch, models[idx], train_datas[idx],
                        optimize=optimizers[idx],
                        phase='training',
                        use_simplified=False,
                        global_model=current_global_model,
                        current_com=com + 1,
                        total_com=args.com,
                        client_mu=mu,
                        args=args,
                        arcface_criterion=arcface_criterion,
                        center_criterion=center_criterion,
                        cross_modal_global_model=None,
                        uncertainty_weighting=uncertainty_weighting
                    )
                    test_loss, test_accuracy, _ = fit_enhanced1(
                        run_id, epoch, models[idx], train_datas[idx],
                        optimize=optimizers[idx],
                        phase='testing',
                        use_simplified=False,
                        global_model=current_global_model,
                        current_com=com + 1,
                        total_com=args.com,
                        client_mu=mu,
                        args=args,
                        arcface_criterion=arcface_criterion,
                        center_criterion=center_criterion,
                        cross_modal_global_model=None,
                        uncertainty_weighting=uncertainty_weighting
                    )

            n = 0
            m = 0
            model1s = [models[0], models[2], models[4]]
            model2s = [models[1], models[3], models[5]]
            prox_server_model1, _ = communication2(n, m, 'fedmylove', prox_server_model1, model1s,
                                                      client_weights)
            prox_server_model2, _ = communication2(n, m, 'fedmylove', prox_server_model2, model2s,
                                                      client_weights)
            if (com + 1) % 10 == 0:
                current_weights = uncertainty_weighting.get_current_weights()
                log_vars = uncertainty_weighting.get_log_vars()
                print(f"\n当前状态 (Com {com + 1})")
                print(
                    f"     学习到的权重 (1/σ²) - CE: {current_weights[0]:.4f}, SupCon: {current_weights[1]:.4f}, FedProx: {current_weights[2]:.4f}, ArcFace: {current_weights[3]:.4f}, Center: {current_weights[4]:.4f}")


            models = [model1s[0], model2s[0], model1s[1], model2s[1], model1s[2], model2s[2]]





        os.makedirs(model_save_dir, exist_ok=True)
        for idx in range(6):
            model_path = os.path.join(model_save_dir, f"model_client_{idx}.pth")
            torch.save(models[idx].state_dict(), model_path)
            print(f"模型{idx} ({names[idx]}) 已保存到: {model_path}")

        # MetaFuse
        for model in models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        weight_predictors[0] = train_weight_network(
            model1=models[0],
            model2=models[1],
            weight_network=weight_predictors[0],
            gallery_file1=train_set_files[0],
            gallery_file2=train_set_files[1],
            args=args,
            batch_size=512
        )


        weight_predictors[1] = train_weight_network(
            model1=models[2],
            model2=models[3],
            weight_network=weight_predictors[1],
            gallery_file1=train_set_files[2],
            gallery_file2=train_set_files[3],
            args=args,
            batch_size=512,
        )


        weight_predictors[2] = train_weight_network(
            model1=models[4],
            model2=models[5],
            weight_network=weight_predictors[2],
            gallery_file1=train_set_files[4],
            gallery_file2=train_set_files[5],
            args=args,
            batch_size=512
        )

        test_with_weight_network(
            model1=models[0],
            model2=models[1],
            weight_network=weight_predictors[0],
            gallery_file1=train_set_files[0],
            gallery_file2=train_set_files[1],
            query_file1=test_set_files[0],
            query_file2=test_set_files[1],
            path_rst='path/to/results',
            roc_save_dir=f"./Results/BLUE-NIR",
            test_name="fedper_weight_network_0"

        )

        test_with_weight_network(
            model1=models[2],
            model2=models[3],
            weight_network=weight_predictors[1],
            gallery_file1=train_set_files[2],
            gallery_file2=train_set_files[3],
            query_file1=test_set_files[2],
            query_file2=test_set_files[3],
            path_rst='path/to/results',
            roc_save_dir=f"./Results/WHT-700",
            test_name="fedper_weight_network_0"

        )

        test_with_weight_network(
            model1=models[4],
            model2=models[5],
            weight_network=weight_predictors[2],
            gallery_file1=train_set_files[4],
            gallery_file2=train_set_files[5],
            query_file1=test_set_files[4],
            query_file2=test_set_files[5],
            path_rst='path/to/results',
            roc_save_dir=f"./Results/TP-TV",
            test_name="fedper_weight_network_0"

        )

        # Testing for single-modal
        print('------------\n')
        avgacc = 0
        avgeer = 0
        print(names[0], '->', names[0])
        test0(models[0], train_set_files[0], test_set_files[0],
              roc_save_dir="./Results/BLUE",
              test_name="BLUE"
              )

        print(names[1], '->', names[1])

        test0(models[1], train_set_files[1], test_set_files[1],
              roc_save_dir=f"./Results/NIR",
              test_name="NIR"
              )
        print(names[2], '->', names[2])

        test0(models[2], train_set_files[2], test_set_files[2],
              roc_save_dir=f"./Results/WHT",
              test_name="WHT"
              )

        print(names[3], '->', names[3])

        test0(models[3], train_set_files[3], test_set_files[3],
              roc_save_dir=f"./Results/700",
              test_name="700"
              )

        print(names[4], '->', names[4])

        test0(models[4], train_set_files[4], test_set_files[4],
              roc_save_dir=f"./Results/TP",
              test_name="TP"
              )

        print(names[5], '->', names[5])

        test0(models[5], train_set_files[5], test_set_files[5],
              roc_save_dir=f"./Results/TV",
              test_name="TV"
              )




