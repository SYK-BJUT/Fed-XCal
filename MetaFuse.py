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
avgeer = 0
avgacc = 0
class SimpleWeightPredictor(nn.Module):


    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleWeightPredictor, self).__init__()


        self.correctness_analyzer = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )


        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim // 4, 2),
            nn.Softmax(dim=-1)
        )

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


        with torch.no_grad():
            final_layer = None
            for m in self.weight_predictor.modules():
                if isinstance(m, nn.Linear) and m.out_features == 2:
                    final_layer = m
                    break

            if final_layer is not None:

                nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)

                nn.init.constant_(final_layer.bias, 0.0)


    def forward(self, output1, output2):
        combined = torch.cat([output1, output2], dim=-1)
        correctness_features = self.correctness_analyzer(combined)
        weights = self.weight_predictor(correctness_features)
        return weights
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
#MetaFuse train
def train_weight_network(model1, model2, weight_network, gallery_file1, gallery_file2, args,batch_size=512):

    print("开始权重网络训练...")

    if os.path.exists('best_weight_network.pth'):
        os.remove('best_weight_network.pth')



    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    trainset1 = MyDataset(txt=gallery_file1, transforms=None, train=True)
    trainset2 = MyDataset(txt=gallery_file2, transforms=None, train=True)

    data_loader_train1 = DataLoader(dataset=trainset1, batch_size=batch_size, num_workers=2, shuffle=False)
    data_loader_train2 = DataLoader(dataset=trainset2, batch_size=batch_size, num_workers=2, shuffle=False)

    print("提取训练数据...")
    output1, _, feat1, targets1 = extract_features_and_outputs(data_loader_train1, model1)
    output2, _, feat2, targets2 = extract_features_and_outputs(data_loader_train2, model2)

    assert len(output1) == len(output2), "两个模型的输出数量不一致"

    if not np.array_equal(targets1, targets2):
        idx1 = np.argsort(targets1)
        idx2 = np.argsort(targets2)
        output1 = output1[idx1]
        output2 = output2[idx2]
        targets1 = targets1[idx1]
        targets2 = targets2[idx2]



    output1_tensor = torch.tensor(output1, dtype=torch.float32, device='cuda')
    output2_tensor = torch.tensor(output2, dtype=torch.float32, device='cuda')
    labels_tensor = torch.tensor(targets1, dtype=torch.long, device='cuda')



    print(f"数据范围: output1=[{output1_tensor.min():.3f}, {output1_tensor.max():.3f}]")
    print(f"        output2=[{output2_tensor.min():.3f}, {output2_tensor.max():.3f}]")

    with torch.no_grad():
        pred1 = torch.argmax(output1_tensor, dim=1)
        pred2 = torch.argmax(output2_tensor, dim=1)
        acc1 = (pred1 == labels_tensor).float().mean().item() * 100
        acc2 = (pred2 == labels_tensor).float().mean().item() * 100
        best_single = max(acc1, acc2)

        print(f"   模型1准确率: {acc1:.2f}%")
        print(f"   模型2准确率: {acc2:.2f}%")

        ideal_weights = calculate_ideal_weights(output1_tensor, output2_tensor, labels_tensor)
        print(f"   理想权重统计: 均值=[{ideal_weights[:, 0].mean():.3f}, {ideal_weights[:, 1].mean():.3f}]")
        print(f"               标准差={ideal_weights[:, 0].std():.3f}")


    weight_network = weight_network.cuda()
    optimizer = torch.optim.Adam(weight_network.parameters(), lr=args.wn_lr, weight_decay=args.wn_weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.wn_step_size, gamma=args.wn_gamma)

    best_acc = 0.0


    for epoch in range(args.wn_epochs):
        weight_network.train()
        optimizer.zero_grad()
        weights = weight_network(output1_tensor, output2_tensor)
        total_loss, supervision_loss = calculate_comprehensive_loss(
            weights, output1_tensor, output2_tensor, labels_tensor
        )
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        weight_network.eval()
        with torch.no_grad():
            fused_output = weights[:, 0:1] * output1_tensor + weights[:, 1:2] * output2_tensor
            fused_pred = torch.argmax(fused_output, dim=1)
            current_acc = (fused_pred == labels_tensor).float().mean().item() * 100
            w1_mean = weights[:, 0].mean().item()
            w1_std = weights[:, 0].std().item()
            ideal_weights = calculate_ideal_weights(output1_tensor, output2_tensor, labels_tensor)
            weight_similarity = F.cosine_similarity(weights, ideal_weights, dim=1).mean().item()
        improvement = current_acc - best_single
        if epoch % 10 == 0 or improvement > 0.5:
            print(
                f"Epoch {epoch:3d}: 监督损失={supervision_loss.item():.4f} 融合ACC={current_acc:.2f}% ({improvement:+.2f}%) ")
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(weight_network.state_dict(), 'best_weight_network.pth')
            if improvement > 0.2:
                print(f"新最佳: {best_acc:.2f}% (提升 {improvement:.2f}%)")

    if os.path.exists('best_weight_network.pth'):
        try:
            weight_network.load_state_dict(torch.load('best_weight_network.pth'))
            final_improvement = best_acc - best_single

            print(
                f"\n最佳融合ACC: {best_acc:.2f}% 性能提升: {final_improvement:+.2f}% ({final_improvement / best_single * 100:+.1f}%)")
        except Exception as e:
            print(f"失败: {e}")

    return weight_network
def calculate_ideal_weights(output1, output2, labels):


    pred1 = torch.argmax(output1, dim=1)
    pred2 = torch.argmax(output2, dim=1)


    correct1 = (pred1 == labels).float()
    correct2 = (pred2 == labels).float()


    total_correct = correct1 + correct2 + 1e-8
    weight1 = correct1 / total_correct
    weight2 = correct2 / total_correct


    both_correct_mask = (correct1 == 1) & (correct2 == 1)
    if both_correct_mask.any():
        prob1 = torch.softmax(output1, dim=1)
        prob2 = torch.softmax(output2, dim=1)


        sorted1, _ = torch.sort(prob1[both_correct_mask], descending=True, dim=1)
        sorted2, _ = torch.sort(prob2[both_correct_mask], descending=True, dim=1)

        quality1 = sorted1[:, 0] - sorted1[:, 1]
        quality2 = sorted2[:, 0] - sorted2[:, 1]


        total_quality = quality1 + quality2 + 1e-8
        weight1[both_correct_mask] = quality1 / total_quality
        weight2[both_correct_mask] = quality2 / total_quality


    both_wrong_mask = (correct1 == 0) & (correct2 == 0)
    if both_wrong_mask.any():
        prob1 = torch.softmax(output1, dim=1)
        prob2 = torch.softmax(output2, dim=1)


        prob1_wrong = prob1[both_wrong_mask]
        prob2_wrong = prob2[both_wrong_mask]
        labels_wrong = labels[both_wrong_mask]


        max_prob1, _ = torch.max(prob1_wrong, dim=1)
        max_prob2, _ = torch.max(prob2_wrong, dim=1)


        true_label_prob1 = prob1_wrong.gather(1, labels_wrong.unsqueeze(1)).squeeze(1)
        true_label_prob2 = prob2_wrong.gather(1, labels_wrong.unsqueeze(1)).squeeze(1)


        quality1 = 1 - (max_prob1 - true_label_prob1)
        quality2 = 1 - (max_prob2 - true_label_prob2)


        quality1 = torch.clamp(quality1, min=0.01, max=0.99)
        quality2 = torch.clamp(quality2, min=0.01, max=0.99)


        total_quality = quality1 + quality2 + 1e-8
        weight1[both_wrong_mask] = quality1 / total_quality
        weight2[both_wrong_mask] = quality2 / total_quality

    return torch.stack([weight1, weight2], dim=1)
def calculate_accuracy_loss_with_preds(weights, output1, output2, labels):


    fused_output = weights[:, 0:1] * output1 + weights[:, 1:2] * output2


    loss = F.cross_entropy(fused_output, labels)

    return loss
def calculate_eer_loss(weights, output1, output2, labels):

    with torch.no_grad():
        ideal_weights = calculate_ideal_weights(output1, output2, labels)

    supervision_loss = F.mse_loss(weights, ideal_weights)
    return supervision_loss
def calculate_comprehensive_loss(weights, output1, output2, labels, alpha=0, beta=1):


    performance_loss = torch.tensor(0.0, device=weights.device, requires_grad=True)
    supervision_loss = calculate_eer_loss(weights, output1, output2, labels)

    total_loss =  supervision_loss

    return total_loss, supervision_loss

#MetaFuse test
def test_with_weight_network(model1, model2, weight_network, gallery_file1, gallery_file2, query_file1, query_file2,
                             path_rst, use_product_reconstruction=0, roc_save_dir=None,
                             test_name="test_with_weight_network"):

    global avgeer, avgacc
    print('Start Testing with Weight Network!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    print(f"使用测试模式: {'乘积重构' if use_product_reconstruction else '原始方法'}")


    path_hard = os.path.join(path_rst, 'rank1_hard')
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)
    if not os.path.exists(path_hard):
        os.makedirs(path_hard)


    trainset1 = MyDataset(txt=gallery_file1, transforms=None, train=True)
    trainset2 = MyDataset(txt=gallery_file2, transforms=None, train=True)
    testset1 = MyDataset(txt=query_file1, transforms=None, train=False)
    testset2 = MyDataset(txt=query_file2, transforms=None, train=False)

    batch_size = 512
    data_loader_train1 = DataLoader(dataset=trainset1, batch_size=batch_size, num_workers=2)
    data_loader_train2 = DataLoader(dataset=trainset2, batch_size=batch_size, num_workers=2)
    data_loader_test1 = DataLoader(dataset=testset1, batch_size=batch_size, num_workers=2)
    data_loader_test2 = DataLoader(dataset=testset2, batch_size=batch_size, num_workers=2)


    print("=== 提取特征和output ===")
    train_output1, _, featDB_train1, iddb_train1 = extract_features_and_outputs(data_loader_train1, model1)
    train_output2, _, featDB_train2, iddb_train2 = extract_features_and_outputs(data_loader_train2, model2)
    test_output1, _, featDB_test1, iddb_test1 = extract_features_and_outputs(data_loader_test1, model1)
    test_output2, _, featDB_test2, iddb_test2 = extract_features_and_outputs(data_loader_test2, model2)


    print("=== 计算匹配分数 ===")
    s1, s2, l, ntest, ntrain = calculate_matching_scores(
        featDB_test1, featDB_test2, featDB_train1, featDB_train2,
        iddb_test1, iddb_test2, iddb_train1, iddb_train2
    )

    if use_product_reconstruction == 0:

        print("=== 使用原始权重网络方法 ===")
        weight_network.eval()
        with torch.no_grad():
            test_output1_tensor = torch.tensor(test_output1, dtype=torch.float32, device='cuda')
            test_output2_tensor = torch.tensor(test_output2, dtype=torch.float32, device='cuda')
            test_weights = weight_network(test_output1_tensor, test_output2_tensor)
            test_weights_np = test_weights.cpu().numpy()


            s1_matrix = s1.reshape(ntest, ntrain)
            s2_matrix = s2.reshape(ntest, ntrain)


            final_scores_matrix = (test_weights_np[:, 0:1] * s1_matrix +
                                   test_weights_np[:, 1:2] * s2_matrix)
            final_scores = final_scores_matrix.flatten()

            print(f"权重统计:")
            print(
                f"  测试集权重1 - 均值: {test_weights_np[:, 0].mean():.4f}, 标准差: {test_weights_np[:, 0].std():.4f}")
            print(
                f"  测试集权重2 - 均值: {test_weights_np[:, 1].mean():.4f}, 标准差: {test_weights_np[:, 1].std():.4f}")

    else:

        print("=== 使用乘积重构权重方法 ===")
        weight_network.eval()
        with torch.no_grad():

            test_output1_tensor = torch.tensor(test_output1, dtype=torch.float32, device='cuda')
            test_output2_tensor = torch.tensor(test_output2, dtype=torch.float32, device='cuda')
            query_weights = weight_network(test_output1_tensor, test_output2_tensor)
            query_weights_np = query_weights.cpu().numpy()  # [ntest, 2]


            train_output1_tensor = torch.tensor(train_output1, dtype=torch.float32, device='cuda')
            train_output2_tensor = torch.tensor(train_output2, dtype=torch.float32, device='cuda')
            gallery_weights = weight_network(train_output1_tensor, train_output2_tensor)
            gallery_weights_np = gallery_weights.cpu().numpy()  # [ntrain, 2]


            s1_matrix = s1.reshape(ntest, ntrain)
            s2_matrix = s2.reshape(ntest, ntrain)


            print("计算乘积重构权重矩阵...")


            raw_w1_matrix = np.outer(query_weights_np[:, 0], gallery_weights_np[:, 0])
            raw_w2_matrix = np.outer(query_weights_np[:, 1], gallery_weights_np[:, 1])


            total_matrix = raw_w1_matrix + raw_w2_matrix + 1e-8
            final_w1_matrix = raw_w1_matrix / total_matrix
            final_w2_matrix = raw_w2_matrix / total_matrix


            final_scores_matrix = final_w1_matrix * s1_matrix + final_w2_matrix * s2_matrix
            final_scores = final_scores_matrix.flatten()


            print(f"权重统计:")
            print(
                f"  查询权重1 - 均值: {query_weights_np[:, 0].mean():.4f}, 标准差: {query_weights_np[:, 0].std():.4f}")
            print(
                f"  查询权重2 - 均值: {query_weights_np[:, 1].mean():.4f}, 标准差: {query_weights_np[:, 1].std():.4f}")
            print(
                f"  库存权重1 - 均值: {gallery_weights_np[:, 0].mean():.4f}, 标准差: {gallery_weights_np[:, 0].std():.4f}")
            print(
                f"  库存权重2 - 均值: {gallery_weights_np[:, 1].mean():.4f}, 标准差: {gallery_weights_np[:, 1].std():.4f}")
            print(f"  配对权重1 - 均值: {final_w1_matrix.mean():.4f}, 标准差: {final_w1_matrix.std():.4f}")
            print(f"  配对权重2 - 均值: {final_w2_matrix.mean():.4f}, 标准差: {final_w2_matrix.std():.4f}")


    print("=== 计算性能指标 ===")
    calculate_rank1_accuracy(iddb_test1, iddb_train1, final_scores, ntest, ntrain, path_rst)


    if not os.path.exists(os.path.join(path_rst, 'veriEER')):
        os.makedirs(os.path.join(path_rst, 'veriEER'))

    with open(os.path.join(path_rst, 'veriEER/scores_VeriEER.txt'), 'w') as f:
        for i in range(len(final_scores)):
            f.write(f"{final_scores[i]} {l[i]}\n")


    pathScore = os.path.join(path_rst, 'veriEER/scores_VeriEER.txt')
    surname = 'scores_VeriEER'
    getGI(pathScore, surname)
    getEER(pathScore, surname)


    if roc_save_dir is not None:
        save_roc_data_and_plot(final_scores, l, roc_save_dir, test_name)

    print('Testing completed!')
    print(f"结果保存在: {path_rst}")
def extract_features_and_outputs(data_loader, model):

    features = []
    outputs = []
    outputs_softmax = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch_id, (datas, target) in enumerate(data_loader):
            data = datas[0].cuda()
            target = target.cuda()


            model_output = model(data, target)
            if isinstance(model_output, tuple):
                output = model_output[0]
            else:
                output = model_output

            codes = model.getFeatureCode(data)


            output_softmax = torch.nn.functional.softmax(output, dim=1)

            outputs.append(output.cpu().numpy())
            outputs_softmax.append(output_softmax.cpu().numpy())  # 新增
            features.append(codes.cpu().numpy())
            targets.append(target.cpu().numpy())


    outputs = np.concatenate(outputs, axis=0)
    outputs_softmax = np.concatenate(outputs_softmax, axis=0)  # 新增
    features = np.concatenate(features, axis=0)
    targets = np.concatenate(targets, axis=0)

    return outputs, outputs_softmax, features, targets
def calculate_rank1_accuracy(iddb_test, iddb_train, avg_scores, ntest, ntrain, path_rst):

    global avgacc
    cnt = 0
    corr = 0

    for i in range(ntest):
        probeID = iddb_test[i]

        dis = np.zeros((ntrain, 1))
        for j in range(ntrain):
            dis[j] = avg_scores[cnt]
            cnt += 1

        idx = np.argmin(dis[:])
        galleryID = iddb_train[idx]

        if probeID == galleryID:
            corr += 1

    rankacc = corr / ntest * 100
    avgacc += rankacc
    print('Rank-1 accuracy: %.3f%%' % rankacc)
def calculate_matching_scores(featDB_test1, featDB_test2, featDB_train1, featDB_train2, iddb_test1, iddb_test2,
                              iddb_train1, iddb_train2):

    s1 = []
    s2 = []
    l = []
    ntest = featDB_test1.shape[0]
    ntrain = featDB_train1.shape[0]

    for i in range(ntest):
        feat1 = featDB_test1[i]
        feat2 = featDB_test2[i]


        for j in range(ntrain):
            cosdis1 = np.dot(feat1, featDB_train1[j])
            dis1 = np.arccos(np.clip(cosdis1, -1, 1)) / np.pi
            s1.append(dis1)
            l.append(1 if iddb_test1[i] == iddb_train1[j] else -1)


        for j in range(ntrain):
            cosdis2 = np.dot(feat2, featDB_train2[j])
            dis2 = np.arccos(np.clip(cosdis2, -1, 1)) / np.pi
            s2.append(dis2)

    return np.array(s1), np.array(s2), np.array(l), ntest, ntrain

def getGI(pathScore, surname):

    pathIn = os.path.dirname(pathScore)
    scorename = os.path.basename(pathScore)

    print('\n')
    print('pathIn: ', pathIn)
    print('scorename: ', scorename)
    print('surname:', surname)

    print('start to load matching scores ...\n')


    pathOut = os.path.join(pathIn, surname)
    if os.path.exists(pathOut) == False:
        os.makedirs(pathOut)


    scores = np.loadtxt(pathScore)

    if scores.size == 0 or scores.ndim == 1:
        print(f"Warning: {pathScore} is empty or contains unexpected data.")

    else:
        inscore = scores[scores[:, 1] == 1, 0]

    inscore = scores[scores[:, 1] == 1, 0]

    outscore = scores[scores[:, 1] == -1, 0]

    maxvin = np.max(inscore)
    minvin = np.min(inscore)

    maxvo = np.max(outscore)
    minvo = np.min(outscore)



    meanvin = np.mean(inscore)
    stdvin = np.std(inscore)

    meanvo = np.mean(outscore)
    stdvo = np.std(outscore)


    samples = 100


    inscore = (inscore - minvin) / (maxvin - minvin) * samples
    outscore = (outscore - minvo) / (maxvo - minvo) * samples


    histin = np.zeros((samples + 1, 1), dtype='int32')
    histo = np.zeros((samples + 1, 1), dtype='int32')


    histin = histin[:, 0]
    histo = histo[:, 0]


    for i in inscore:
        i = int(round(i))
        histin[i] += 1
    for i in outscore:
        i = int(round(i))
        histo[i] += 1

    histin = np.array(histin)
    histo = np.array(histo)

    sumtmp = np.sum(histin)
    histin = histin / sumtmp * 100

    sumtmp = np.sum(histo)
    histo = histo / sumtmp * 100

    plt.figure(1)

    plt.plot(np.linspace(0, 1, samples + 1) * (maxvo - minvo) + minvo, histo, 'r', label='Impostor')

    plt.plot(np.linspace(0, 1, samples + 1) * (maxvin - minvin) + minvin, histin, 'b', label='Genuine')

    plt.legend(loc='upper right', fontsize=13)
    plt.xlabel('Matching Score', fontsize=13)
    plt.ylabel('Percentage (%)', fontsize=13)

    plt.ylim([0, 1.2 * np.max([histin.max(), histo.max()])])  # 设置y轴的上限为最大值的1.2倍
    plt.grid(True)

    plt.savefig(os.path.join(pathOut, 'GI_curve.png'))


    with open(os.path.join(pathOut, 'matching_score_distr.txt'), 'w') as f:

        f.writelines('[min, max] [mean +- std]\n')

        f.writelines('inner: [%.10f, %.10f] [%.10f +- %.10f]\n' % (minvin, maxvin, meanvin, stdvin))

        f.writelines('outer: [%.10f, %.10f] [%.10f +- %.10f]\n' % (minvo, maxvo, meanvo, stdvo))
        f.writelines('number of genuine matching:  %d\n' % inscore.shape)
        f.writelines('number of impostor matching: %d\n' % outscore.shape)


    xin = np.linspace(0, 1, samples + 1) * (maxvin - minvin) + minvin
    xo = np.linspace(0, 1, samples + 1) * (maxvo - minvo) + minvo

    with open(os.path.join(pathOut, 'matching_hist.txt'), 'w') as f:

        for i in range(samples + 1):
            f.writelines('%.4f %.4f %.4f %.4f\n' % (xin[i], histin[i], xo[i], histo[i]))

    print('done!\n')
    # 检查命令行参数
def getEER(pathScore, surname):
    global avgeer, avgacc
    pathIn = os.path.dirname(pathScore)
    scorename = os.path.basename(pathScore)

    pathOut = os.path.join(pathIn, surname)
    if os.path.exists(pathOut) == False:
        os.makedirs(pathOut)

    scores = np.loadtxt(pathScore)

    inscore = scores[scores[:, 1] == 1, 0]
    outscore = scores[scores[:, 1] == -1, 0]

    print('scores loading done!\n')

    print('start to calculate EER ...')
    start = time.time()

    print('numbers of inner & outer matching:')
    print(inscore.shape, outscore.shape)

    mIn = inscore.mean()
    mOut = outscore.mean()
    if mIn < mOut:
        inscore = -inscore
        outscore = -outscore

    y = np.vstack((np.ones((len(inscore), 1)), np.zeros((len(outscore), 1))))
    scores = np.vstack((inscore.reshape(-1, 1), outscore.reshape(-1, 1)))

    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    if mIn < mOut:
        thresh = -thresh
        thresholds = -thresholds

    print('eer: %.6f%% th: %.3f auc: %.10f' % (eer * 100, thresh, roc_auc))
    avgeer += eer
    diffV = np.abs(fpr - (1 - tpr))
    idx = np.argmin(diffV)
    eer_1_2 = (fpr[idx] + (1 - tpr[idx])) / 2.0
    th_1_2 = thresholds[idx]
    print('eer_1/2: %.6f%% th_1/2: %.3f auc: %.10f' % (eer_1_2 * 100, th_1_2, roc_auc))

    with open(os.path.join(pathOut, 'rst_eer_th_auc.txt'), 'w') as f:
        f.writelines('%.10f %.4f %.10f\n' % (eer, thresh, roc_auc))
        f.writelines('%.10f %.4f %.10f\n' % (eer_1_2, th_1_2, -1))

    with open(os.path.join(pathOut, 'DET_th_far_frr.txt'), 'w') as f:
        fnr = 1 - tpr
        for i in range(len(fpr)):
            f.writelines('%.6f\t%.10f\t%.10f\n' % (thresholds[i], fpr[i], fnr[i]))

    pdf = PdfPages(os.path.join(pathOut, 'roc_det.pdf'))
    fpr = fpr * 100
    tpr = tpr * 100
    fnr = fnr * 100

    plt.figure()
    plt.plot(fpr, tpr, color='b', linestyle='-', marker='^', label='ROC curve')
    plt.plot(np.linspace(0, 100, 101), np.linspace(100, 0, 101), 'k-', label='EER')

    plt.xlim([0, 5])
    plt.ylim([90, 100])

    plt.legend(loc='best')
    plt.grid(True)
    plt.title('ROC curve')
    plt.xlabel('FAR (%)')
    plt.ylabel('GAR (%)')
    plt.savefig(os.path.join(pathOut, 'ROC.png'))

    pdf.savefig()

    plt.figure()
    plt.plot(fpr, fnr, color='b', linestyle='-', marker='^', label='DET curve')
    plt.plot(np.linspace(0, 100, 101), np.linspace(0, 100, 101), 'k-', label='EER')

    plt.xlim([0, 5])
    plt.ylim([0, 5])

    plt.legend(loc='best')
    plt.grid(True)
    plt.title('DET curve')
    plt.xlabel('FAR (%)')
    plt.ylabel('FRR (%)')
    plt.savefig(os.path.join(pathOut, 'DET.png'))

    pdf.savefig()

    plt.figure()
    plt.plot(thresholds, fpr, color='r', linestyle='-', marker='.', label='FAR')
    plt.plot(thresholds, fnr, color='b', linestyle='-', marker='^', label='FRR')

    plt.legend(loc='best')
    plt.grid(True)
    plt.title('FAR and FRR Curves')
    plt.xlabel('Thresholds')
    plt.ylabel('FAR, FRR (%)')
    plt.savefig(os.path.join(pathOut, 'FAR_FRR.png'))

    pdf.savefig()
    pdf.close()

    print('done')

