#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 08:10:54 2024

@author: adelino
"""
import numpy as np
# -----------------------------------------------------------------------------
def multi_dist(xx,yy):
    # dist = []
    # for sx in xx:
    #     for sy in yy: 
    #         dist.append(np.sqrt((sx-sy)**2))
    # return np.mean(dist)
    Nx = len(xx)
    Ny = len(yy)
    vxx = np.var(xx)
    vyy = np.var(yy)
    return np.sqrt(((Nx-1)*vxx + (Ny-1)*vyy) /(Nx+Ny-2))
# -----------------------------------------------------------------------------
def mean_dist(xx,yy):
    # dist = []
    # for sx in xx:
    #     for sy in yy: 
    #         dist.append(np.sqrt((sx-sy)**2))
    # return np.mean(dist)
    Nx = np.mean(xx)
    Ny = np.mean(yy)
    return np.abs(Nx-Ny)
# -----------------------------------------------------------------------------
def mahalanobis_1d(xx,yy):
    Nx = len(xx)
    Ny = len(yy)
    mxx = np.mean(xx)
    myy = np.mean(yy)
    vxx = np.var(xx)
    vyy = np.var(yy)
    S = np.sqrt(((Nx-1)*vxx + (Ny-1)*vyy) /(Nx+Ny-2))
    return abs(mxx - myy)/S
# -----------------------------------------------------------------------------
