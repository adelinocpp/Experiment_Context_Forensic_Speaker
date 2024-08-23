#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 08:42:32 2024

@author: adelino
"""
import dill
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import pandas as pd

fpx = 'AVT02'
cm = 1/2.54  # centimeters in inches
DATA_FILE = '../PreProcessamento/csvLeeData_02_05_2024.csv'

df_VOGAIS = pd.read_csv(DATA_FILE,sep='\t')
lstLocutores = np.array(df_VOGAIS["Locutor"].unique())
lstFeatures = df_VOGAIS.columns[2:28]

lstLocSex = []
lstLocSexTag = []
for Loc in lstLocutores:
    df_VogLoc = df_VOGAIS[(df_VOGAIS["Locutor"] == Loc)]
    valSex = df_VogLoc.loc[df_VogLoc.first_valid_index(),'Sexo']
    lstLocSexTag.append(valSex.replace("'",""))
    lstLocSex.append(df_VogLoc.loc[df_VogLoc.first_valid_index(),'Sexo'] == "'F'")

tagPrCmpS = []
for x in range(0,18):
    a = r'^{:}'.format(lstLocSexTag[x])
    b = 'Locutor$%s$'%a
    tagPrCmpS.append('{:} {:02d}'.format(b,lstLocutores[x]))

idxSortLoc = np.argsort(lstLocSex)

BOOTSTRAP_SPK_SAMPLE_FILE = '{:}_dict_BootStrap_Speaker_Sample_File.pth'.format(fpx)

with open(BOOTSTRAP_SPK_SAMPLE_FILE, "rb") as dill_file:
    dictBootStrapSpkSample = dill.load(dill_file)  

lstSMFLocsTrain = dictBootStrapSpkSample["SMF_Train"]
lstSMFLocsTest = dictBootStrapSpkSample["SMF_Test"]
lstPCALocsTrain = dictBootStrapSpkSample["PCA_Train"]
lstPCALocsTest = dictBootStrapSpkSample["PCA_Test"]
lstRAWLocsTrain =  dictBootStrapSpkSample["RAW_Train"] 
lstRAWLocsTest = dictBootStrapSpkSample["RAW_Test"]

mxtTestSMF = np.array([])
mxtTestPCA = np.array([])
mxtTestRAW = np.array([])
indexLoc = np.array([])
for i in idxSortLoc: #range(0,len(lstSMFLocsTest)):
    if( i == 0):
        mxtTestSMF = lstSMFLocsTest[i]    
        mxtTestPCA = lstPCALocsTest[i]    
        mxtTestRAW = lstRAWLocsTest[i]    
        indexLoc = i*np.ones((len(lstRAWLocsTest[i]),1))
    else:
        mxtTestSMF = np.vstack((mxtTestSMF,lstSMFLocsTest[i]))    
        mxtTestPCA = np.vstack((mxtTestPCA,lstPCALocsTest[i]))    
        mxtTestRAW = np.vstack((mxtTestRAW,lstRAWLocsTest[i]))    
        indexLoc =  np.vstack((indexLoc,i*np.ones((len(lstRAWLocsTest[i]),1))))

embedding = MDS(n_components=2, normalized_stress='auto')
SMF_MDS = embedding.fit_transform(mxtTestSMF)
PCA_MDS = embedding.fit_transform(mxtTestPCA)
RAW_MDS = embedding.fit_transform(mxtTestRAW)


# vIdxLoc = np.array([nLoc for nLoc in range(0,len(lstLocSex))])
# idx = np.argsort(lstLocSex)


# lstLocSex = np.array(lstLocSex)[idx]
# vIdxLoc = vIdxLoc[idx]
# vIdxSort = []
# for i in vIdxLoc:
#     idxL = np.where(indexLoc == i)[0]
#     vIdxSort = [*vIdxSort, *idxL]

# SMF_MDS = SMF_MDS[vIdxSort,:]
# PCA_MDS = PCA_MDS[vIdxSort,:]
# RAW_MDS = RAW_MDS[vIdxSort,:]

plot_file_name = "MDS_01.jpg"
fig = plt.figure(figsize =(30*cm, 30*cm))
gs = fig.add_gridspec(nrows=2, ncols=2)
ax = gs.subplots(sharex=False,sharey=False)

vColor = ['tab:blue','tab:red', 'tab:green','tab:orange', 
          'tab:purple', 'tab:olive', 'tab:cyan','tab:brown',
          'tab:pink']
vColorM = ['red', 'orange', 'gold', 'darkkhaki','brown']
vColorF = ['blue','purple', 'cyan', 'teal', 'green']
vMarkerM = ['X','P','*']
vMarkerF = ['s','o','D']
for i in range(0,18):
    idxL = np.where(indexLoc == i)[0]
    if (lstLocSex[i]):
        vColor = vColorF
        vMarker = vMarkerF
        sAlpha = 1
    else:
        vColor = vColorM
        vMarker = vMarkerM
        sAlpha = 0.45
    iC = i % len(vColor)
    iM = i % len(vMarker)
   
    ax[0,0].scatter(RAW_MDS[idxL,0], RAW_MDS[idxL,1], color = vColor[iC], 
                marker=vMarker[iM], alpha=sAlpha, label=tagPrCmpS[i])
    ax[0,1].scatter(SMF_MDS[idxL,0], SMF_MDS[idxL,1], color = vColor[iC], 
                marker=vMarker[iM], alpha=sAlpha)
    ax[1,0].scatter(PCA_MDS[idxL,0], PCA_MDS[idxL,1], color = vColor[iC], 
                marker=vMarker[iM], alpha=sAlpha)
    
handles, labels = ax[0,0].get_legend_handles_labels()
# plt.title('Entropia cruzada empírica doconjunto de teste $%s$'%a)
ax[1,1].axis('off')
ax[1,1].legend(handles, labels, loc='upper center', ncols=2)
ax[0,0].set_xlabel('VA dimensão MDS 00')
ax[0,0].set_ylabel('VA dimensão MDS 01')
ax[0,0].grid(color="grey",linewidth=0.5, linestyle='-.')
ax[0,1].set_xlabel('GLM-RES dimensão MDS 0')
ax[0,1].set_ylabel('GLM-RES dimensão MDS 1')
ax[0,1].grid(color="grey",linewidth=0.5, linestyle='-.')
ax[1,0].set_xlabel('PCA dimensão MDS 0')
ax[1,0].set_ylabel('PCA dimensão MDS 1')
ax[1,0].grid(color="grey",linewidth=0.5, linestyle='-.')
# plt.legend(ncols=3)
plt.suptitle('Dispersão do conjunto de testes no espaço MDS',y=0.91,fontsize=20)
fig.savefig(plot_file_name,dpi=300,bbox_inches='tight')  


with open("dict_SMF.pth", "rb") as dill_file:
    dictLR_SMF = dill.load(dill_file)  
with open("dict_PCA.pth", "rb") as dill_file:
    dictLR_PCA = dill.load(dill_file)  
with open("dict_RAW.pth", "rb") as dill_file:
    dictLR_RAW = dill.load(dill_file)  