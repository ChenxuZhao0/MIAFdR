import logging
import os
import datetime
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Subset, ConcatDataset
from scipy import interpolate
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix
from statsmodels.stats.multitest import multipletests


from utils.model import MLPClassifier
from pretrain import evaluate


def cal_fdr(labels, preds):
    upper = 0
    under = 0
    for idx in range(len(labels)):
        if labels[idx] == 0 and preds[idx] == 1:
            upper += 1
        if preds[idx] == 1:
            under += 1
    under = max(1, under)
    fdr = upper / under
    return fdr


def cal_auc(fprs, tprs):
    auc = 0
    for i in range(1, len(tprs)):
        auc += (fprs[i] - fprs[i-1]) * (tprs[i] + tprs[i-1]) / 2
    return auc



def evaluate_classifier(logging, target_net, attack_model, test_non_member_loader, test_member_loader, device):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    pred_n = 0
    pred_p = 0

    pred_probs = []
    true_labels = []

    # collect prediction scores for test data on attack model
    # get prediction results for test data
    with torch.no_grad():
        for ind, data in enumerate(test_non_member_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = target_net(inputs)

            outputs = torch.nn.functional.softmax(outputs, dim=1)

            scores = attack_model(outputs)
            predicted = torch.round(scores)
            # _, predicted = torch.max(scores.data, 1)
            pred_n += (predicted == 1).sum().item()
            pred_p += (predicted == 0).sum().item()
            TN += (predicted == 1).sum().item()
            FP += (predicted == 0).sum().item()

            pred_probs.extend(scores.cpu().detach().numpy())
            true_labels.extend(np.ones((scores.shape[0], 1)))

    with torch.no_grad():
        for ind, data in enumerate(test_member_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = target_net(inputs)

            outputs = torch.nn.functional.softmax(outputs, dim=1)

            scores = attack_model(outputs)
            predicted = torch.round(scores)
            # _, predicted = torch.max(scores.data, 1)
            pred_n += (predicted == 1).sum().item()
            pred_p += (predicted == 0).sum().item()
            TP += (predicted == 0).sum().item()
            FN += (predicted == 1).sum().item()
            pred_probs.extend(scores.cpu().detach().numpy())
            true_labels.extend(np.zeros((scores.shape[0], 1)))

    FPR_list, TPR_list, Thres_list = roc_curve(true_labels, pred_probs)
    auc_classifier = cal_auc(FPR_list, TPR_list)

    if TP + FP == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)

    balenced_accuracy = (TP + TN) / (TP + TN + FP + FN)

    return balenced_accuracy, Precision, auc_classifier, TPR_list, FPR_list, Thres_list


def get_results(logging, n_non_member, p_values, alpha):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    rejects, _, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')  # reject true means member

    for i in range(len(rejects)):
        if rejects[i] == False:
            if i < n_non_member:
                TN += 1
            else:
                FN += 1
        else:
            if i < n_non_member:
                FP += 1
            else:
                TP += 1

    if FP + TP == 0:
        FDR = 0
    else:
        FDR = FP / (FP + TP)

    if TP + FP == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    balenced_accuracy = (TP + TN) / (TP + TN + FP + FN)

    return balenced_accuracy, TPR, FPR, FDR, Precision


def evaluate_ours(args, logging, alphas_v, n_non_member, p_values):
    TPR_list = []
    FPR_list = []
    FDR_list = []
    balenced_accuracy_list = []
    Precision_list = []

    for alpha in alphas_v:
        balenced_accuracy, TPR, FPR, FDR, Precision = get_results(logging, n_non_member, p_values, 1/args.alpha * alpha)
        TPR_list.append(TPR)
        FPR_list.append(FPR)
        FDR_list.append(FDR)
        balenced_accuracy_list.append(balenced_accuracy)
        Precision_list.append(Precision)

    auc_ours = cal_auc(FPR_list, TPR_list)

    return TPR_list, FPR_list, FDR_list, balenced_accuracy_list, auc_ours, Precision_list
