#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:11:33 2023

@author: adelino
"""
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

# -----------------------------------------------------------------------------
def cllr(x,sample_SO,kde_SO,sample_DO,kde_DO):
    LR_SO = interp1d(x,kde_SO)(sample_SO)/interp1d(x,kde_DO)(sample_SO)
    LR_DO = interp1d(x,kde_SO)(sample_DO)/interp1d(x,kde_DO)(sample_DO)
    LRSO = np.mean(np.log2(1+ 1/LR_SO))
    LRDO = np.mean(np.log2(1 + LR_DO))
    return 0.5*(LRSO + LRDO), LR_SO, LR_DO
# -----------------------------------------------------------------------------
def norm_ZO(x):
    minx = np.min(x)
    maxx = np.max(x)
    return (x-minx)/(maxx - minx) + 1e-32
# -----------------------------------------------------------------------------
def ece_curve(x,sample_SO,kde_SO,sample_DO,kde_DO,x_PR=None):
    LR_SO = interp1d(x,kde_SO)(sample_SO)/interp1d(x,kde_DO)(sample_SO)
    LR_DO = interp1d(x,kde_SO)(sample_DO)/interp1d(x,kde_DO)(sample_DO)
    if (x_PR is None):
        x_PR = np.logspace(-2.5, 2.5, num=400, base=10)
    ECE = np.zeros(x_PR.shape)
    ECE_ref = np.zeros(x_PR.shape)
    for i, x in enumerate(x_PR):
        LRSO = (x/(1+x))*np.mean(np.log2(1+ 1/(LR_SO*x)))
        LRDO = (1/(1+x))*np.mean(np.log2(1 + LR_DO*x))
        ECE[i] = LRSO + LRDO
        ECE_ref[i] = (x/(1+x))*np.log2(1+ 1/x) + (1/(1+x))*np.log2(1 + x)
                 
    return x_PR, ECE, ECE_ref
# -----------------------------------------------------------------------------
def tippet(pred_Y, score_X, num=400):
    fpr, tpr, thresholds = roc_curve(pred_Y, score_X, pos_label=0)
    idx_z = (pred_Y == 0).nonzero()[0]
    idx_o = (pred_Y == 1).nonzero()[0]
    scr_z = score_X[idx_z]
    scr_o = score_X[idx_o]
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    bas_thr = (np.max(score_X) - np.min(score_X))/num
    thresholds = np.concatenate((np.linspace(np.min(score_X),thresh-bas_thr,int(num/2)),
                                np.linspace(thresh+bas_thr,np.max(score_X),int(num/2))))
    fpr = np.zeros((num,))
    tpr = np.zeros((num,))
    
    np.sum((pred_Y == 1))
    for idx, thr in enumerate(thresholds):
        TP = np.sum((scr_o > thr))
        FP = np.sum((scr_z > thr))
        TN = np.sum((scr_z < thr))
        FN = np.sum((scr_o < thr))
        
        fpr[idx] = FP/(TN + FP)
        tpr[idx] = TP/(TP + FN)
    return fpr, tpr, thresholds
# -----------------------------------------------------------------------------