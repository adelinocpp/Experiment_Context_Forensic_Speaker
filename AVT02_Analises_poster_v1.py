#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:47:44 2024

@author: adelino
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.phonokepping import ajustVowelTag, FoneticaToTable
from scipy.stats import f_oneway, tukey_hsd, gaussian_kde, t, sem # norm
from scipy.special import expit
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from fsc_utils import ece_curve #cllr, norm_ZO
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from utils.phonokepping import soundBinaryClass # vowelBinaryClass, consonantBinaryClass
import dill
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
# -----------------------------------------------------------------------------
def anovaResult(lstMedidas,tagLabel,mfileName):
    tagsComp = []
    limitsComp = []
    nVSn = []
    anova = f_oneway(*lstMedidas)
    result = tukey_hsd(*lstMedidas)
    conf = result.confidence_interval(confidence_level=.95)
    for ((i, j), l) in np.ndenumerate(conf.low):
        # filter out self comparisons
        if (i != j) and (j> i):
            h = conf.high[i,j]
            tagsComp.append("{:} vs. {:}".format(tagLabel[i],tagLabel[j]))
            limitsComp.append([l,h])
            nVSn.append(l*h)  
    valorp = anova.pvalue    
    with open(mfileName, "w") as f:
        print("Resultado Anova", file=f)
        print("Valor-p: {:}".format(anova.pvalue), file=f)
        print("Estatistica: {:}".format(anova.statistic), file=f)
        if (anova.pvalue < 0.05):
            print("Rejeita H0: pelo menos uma média é diferente",file=f)
        else:
            print("Falha em rejeitar H0: sem evidência para afirmar que uma média é diferente",file=f)
        for i in range(0,len(tagsComp)):
            print("{:}: low: {:.2f}; high: {:.2f}".format(tagsComp[i],limitsComp[i][0],limitsComp[i][1]),file=f)
    return tagsComp, limitsComp,  nVSn, valorp 
# =============================================================================
def withinSetDistance(setData):
    lstDeviation = []
    for subSetData in setData:
        pCenter = np.mean(subSetData,axis=0)
        lstDistance = []
        for sampleData in subSetData:
            lstDistance.append(np.linalg.norm(sampleData - pCenter))    
        lstDeviation.append(np.mean(lstDistance))
    return lstDeviation
# =============================================================================
def betweenSetDistance(setData):
    lstFullSet = []
    for subSetData in setData:
        for sampleData in subSetData:
            lstFullSet.append(sampleData)    
    pCenter = np.mean(lstFullSet,axis=0)        
    subSetDistance = []
    for subSetData in setData:
        subSetCenter = np.mean(subSetData,axis=0)    
        subSetDistance.append(np.linalg.norm(subSetCenter - pCenter))   
        
    return subSetDistance    
# =============================================================================
def ratioDispersionSet(setData):
   lstWi = withinSetDistance(setData)
   lstBw = betweenSetDistance(setData)
   locRatio = np.divide(lstWi,lstBw/np.max(lstBw))
   return np.mean(lstWi)/np.mean((lstBw/np.max(lstBw))), np.std(locRatio)/np.sqrt(len(locRatio))
# =============================================================================
def buildTable(dictData, lstVertLabel, filename):
    lstKeys = dictData.keys()
    nKeys = len(lstKeys)
    nRow = len(lstVertLabel)
    mtxValores = np.zeros((nRow+1,nKeys+1))
    for i in range(0,nRow):
        for k, valK in enumerate(lstKeys):
            mtxValores[i,k] = dictData[valK][i]
        mtxValores[i,k+1] =  np.sum(mtxValores[i,0:k+1])
    
    for k in range(0,nKeys+1):
        mtxValores[i+1,k] =  np.sum(mtxValores[0:i+1,k])
    
    lstMatrix = []
    for i in range(0,nRow+1):
        lstLine = []
        for k in range(0,nKeys+1):
            # mtxValores[i,k] = "{:.0f} ({:.1f})".format(mtxValores[i,k],100*mtxValores[i,k]/mtxValores[i,-1]).replace(".",",")
            lstLine.append("{:.0f} ({:.1f})".format(mtxValores[i,k],100*mtxValores[i,k]/mtxValores[i,-1]).replace(".",","))
        lstMatrix.append(lstLine)        
            
    colNames = [*lstKeys,'Total']
    csvData = pd.DataFrame(lstMatrix,columns=colNames)
    csvData.index =[*["Locutor {:02}".format(Loc) for Loc in lstVertLabel], 'Total']
    csvData.to_csv(filename,sep='\t')
# =============================================================================
def sampleTrainTestOFRandonMeans(mxtData,pTrain = 0.7, tSample = 20, mSize=0.1):
    nSample, nFeat = mxtData.shape
    nTrain = int(pTrain*nSample)
    nMean  = int(mSize*nSample)
    rPerm  = np.random.permutation(nSample)
    mtxTrain = mxtData[rPerm[:nTrain],:]
    mtxTest  = mxtData[rPerm[nTrain:],:]
    nSTrain = int(pTrain*tSample)
    nSTest = int((1-pTrain)*tSample)
    sMtxTrain = np.zeros((nSTrain,nFeat))
    sMtxTest = np.zeros((nSTest,nFeat))
    for j in range(nSTrain):
        rPerm  = np.random.permutation(mtxTrain.shape[0])
        sMtxTrain[j,:] = np.mean(mtxTrain[rPerm[:nMean],:],axis=0)
    for j in range(nSTest):
        rPerm  = np.random.permutation(mtxTest.shape[0])
        sMtxTest[j,:] = np.mean(mtxTest[rPerm[:nMean],:],axis=0)
    return sMtxTrain, sMtxTest

# =============================================================================
def resampleTrainTestOFRandonMeans(mtxTrain,mtxTest, pTrain = 0.7, tSample = 20, mSize=0.2):
    nSampleTr, nFeat = mtxTrain.shape
    nSampleTs, nFeat = mtxTest.shape
    nSample = nSampleTs + nSampleTr
    nMean  = int(mSize*nSample)
    nSTrain = int(pTrain*tSample)
    nSTest = int((1-pTrain)*tSample)
    sMtxTrain = np.zeros((nSTrain,nFeat))
    sMtxTest = np.zeros((nSTest,nFeat))
    for j in range(nSTrain):
        rPerm  = np.random.permutation(mtxTrain.shape[0])
        sMtxTrain[j,:] = np.mean(mtxTrain[rPerm[:nMean],:],axis=0)
    for j in range(nSTest):
        rPerm  = np.random.permutation(mtxTest.shape[0])
        sMtxTest[j,:] = np.mean(mtxTest[rPerm[:nMean],:],axis=0)
    return sMtxTrain, sMtxTest

# =============================================================================
def sampleTrainTestFromDataFrame(dftData,SAMPLE_FILE, pVariables, RESAMPLE_DATA=False, TRAIN_SL = 0.7):
    nLevels = []
    for var in pVariables:
        nLevels.append(len(dftData[var].unique()))
                       
    dictSample = {}
    locTrainSample = []
    locTestSample = []
    lstLocutoresTMP = np.array(dftData["Locutor"].unique())
    if not os.path.exists(SAMPLE_FILE) or RESAMPLE_DATA: 
        for loc in lstLocutoresTMP:
            idxLoc = (np.array(dftData["Locutor"]) == loc).nonzero()[0]
            nObs = len(idxLoc)
            permIdx = np.random.permutation(nObs)
            idxTr = int(np.floor(TRAIN_SL*nObs))
            locTrainSample.append(idxLoc[permIdx[:idxTr]])
            locTestSample.append(idxLoc[permIdx[idxTr:]])
            
        dictSample["train"] = locTrainSample
        dictSample["test"] = locTestSample
        with open(SAMPLE_FILE, "wb") as dill_file:
            dill.dump(dictSample, dill_file)
    else:
        with open(SAMPLE_FILE, "rb") as dill_file:
            dictSample = dill.load(dill_file)   
        locTrainSample = dictSample["train"]
        locTestSample = dictSample["test"]

    idxTrain = np.array([])
    idxTest = np.array([])
    for idx in locTrainSample:
        idxTrain = np.hstack((idxTrain,idx))
    for idx in locTestSample:
        idxTest = np.hstack((idxTest,idx))
    
    dfTrain = dftData.iloc[idxTrain,:]
    boolLevel = True
    for k, var in enumerate(pVariables):
        boolLevel = boolLevel and (nLevels[k] == len(dfTrain[var].unique()))
    
    if (boolLevel):
        return idxTrain, idxTest
    else:
        print("loop")
        return sampleTrainTestFromDataFrame(dftData,SAMPLE_FILE, pVariables, True, TRAIN_SL)
# =============================================================================
def runComparison(lstLocs):
    lstSameLoc = []
    lstScore = []
    nLoc = len(lstLocs)
    for i in range(0,nLoc):
        mxtFeatI = lstLocs[i]
        for j in range(i,nLoc):
            mxtFeatJ = lstLocs[j]
            if (i == j):
                for k in range(0,mxtFeatI.shape[0]-1):
                    for l in range(k+1,mxtFeatJ.shape[0]):
                        lstScore.append(np.linalg.norm(mxtFeatI[k,:] - mxtFeatJ[l,:]))
                        lstSameLoc.append(1)
            else:
                for k in range(0,mxtFeatI.shape[0]):
                    for l in range(0,mxtFeatJ.shape[0]):
                        lstScore.append(np.linalg.norm(mxtFeatI[k,:] - mxtFeatJ[l,:]))
                        lstSameLoc.append(0)
    return lstScore, lstSameLoc 
# =============================================================================
def runLogisticRegression(lstScoreTrain,lstSameLocTrain,lstScoreLocTest,lstSameLocTest):
    xScore = np.array(lstScoreTrain).ravel().reshape(-1, 1)
    # xScore.shape = (len(xScore),)
    yPred = np.array(lstSameLocTrain).ravel().reshape(-1, 1)
    yPred.shape = (len(xScore),)
    clf_res = LogisticRegression(random_state=0, 
                                 class_weight = 'balanced').fit(xScore,yPred)
    y_Log_prob = clf_res.predict_log_proba(np.array(lstScoreLocTest).reshape(-1, 1))
    LLR_res = y_Log_prob[:,1]-y_Log_prob[:,0]
    stdx = np.std(LLR_res)
    minx = np.min(LLR_res)
    maxx = np.max(LLR_res)
    # --- densidades
    x_sup = np.linspace(minx-0.25*stdx,maxx+0.25*stdx,400)
    idxSS = (np.array(lstSameLocTest) == 1).nonzero()[0]
    idxDS = (np.array(lstSameLocTest) == 0).nonzero()[0]
    SS_LLR = LLR_res[idxSS]
    SS_y = gaussian_kde(SS_LLR, bw_method=0.25).pdf(x_sup)
    DS_LLR =  LLR_res[idxDS]
    DS_y = gaussian_kde(DS_LLR, bw_method=0.25).pdf(x_sup)

    x_PR, ECE, ECE_ref = ece_curve(x_sup,SS_LLR,SS_y,DS_LLR,DS_y)
    cllr_n = np.max(ECE)
    fpr, tpr, thresholds = roc_curve(lstSameLocTest, LLR_res, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    dictReturn = {}
    dictReturn["ECE_x"] = x_PR
    dictReturn["ECE_y"] = ECE
    dictReturn["ECE_r"] = ECE_ref
    dictReturn["CLLR"] = cllr_n
    dictReturn["EER"] = eer
    dictReturn["THR"] = thresh
    dictReturn["LOGISTIC_REG"] = clf_res
    dictReturn["XSUP"] = x_sup
    dictReturn["PDF_SS"] = SS_y
    dictReturn["PDF_DS"] = DS_y
    return dictReturn
# =============================================================================
def plotLogisticResults(dictData,baseFigName):
    
    idxSameLoc = np.where(np.array(dictData['SameLoc_Test']) == 1)[0]
    idxDiffLoc = np.where(np.array(dictData['SameLoc_Test']) == 0)[0]
    clf = dictData['LOGISTIC_REG']
    LLRData = clf.predict_log_proba(np.array(dictData['Score_Test']).reshape(-1, 1))
    LLRRatio =  LLRData[:,1]-LLRData[:,0]
    same_LOC_data = LLRRatio[idxSameLoc]
    diff_LOC_data = LLRRatio[idxDiffLoc]
    same_LOC_pred = np.ones((len(same_LOC_data),))
    diff_LOC_pred = np.zeros((len(diff_LOC_data),))
    # pred_prob = clf.predict_proba(dictData['XSUP'].reshape(-1, 1))
    x_sup_clf = np.linspace(np.min(dictData['XSUP']),np.max(dictData['XSUP']),400).reshape(-1, 1)
    y_sup_clf = 1- expit(x_sup_clf * clf.coef_ + clf.intercept_).ravel()
    
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25.5*cm,12*cm))
    
    ax1.hist(same_LOC_data, 25, density=True, histtype='stepfilled', color='b',alpha=0.5, label = "mesmo locutor")
    ax1.plot(dictData['XSUP'],dictData['PDF_SS'], color = "b", linewidth=2)
    ax1.hist(diff_LOC_data, 25, density=True, histtype='stepfilled', color='r',alpha=0.5, label = "locutor diferente")
    ax1.plot(dictData['XSUP'],dictData['PDF_DS'], color = "r",linewidth=2)
    ax1.set_title('EER = {:4.1f}% - CLLR = {:5.3f}'.format(100*dictData['EER'],dictData['CLLR']))
    ax1.set_xlabel('Pontuação da comparação')
    ax1.set_ylabel('densidade de ocorrências')
    ax1.grid(color='tab:gray', linestyle='-.', linewidth=0.5)
    ax1.legend()
    # fig.savefig("{:}_Density.png".format(baseFigName),dpi=200,bbox_inches='tight')
    
    
    
    # fig = plt.figure(figsize =(8*cm,8*cm))
    ax2.plot(x_sup_clf,y_sup_clf, linestyle='-', color='g', linewidth=2,label="regressão")
    ax2.scatter(same_LOC_data, same_LOC_pred, color='b', linewidth=2,label="mesmo locutor", alpha=0.1)
    ax2.scatter(diff_LOC_data, diff_LOC_pred, color='r', linewidth=2,label="locutor diferente", alpha=0.1)
    ax2.grid(color='tab:gray', linestyle='-.', linewidth=0.5)
    ax2.set_title('Regressão logística de separação')
    ax2.set_xlabel('pontuação de seperação')
    ax2.set_ylabel('Probabilidade de ser o mesmo locutor')
    ax2.legend()
    fig.savefig("{:}_Logistic.png".format(baseFigName),dpi=200,bbox_inches='tight')

# =============================================================================
def leaveOneOutJack(dictData, lstLocData, nRep=90):
    ACC = []
    FPR = []
    FNR = []
    for i in range(0,nRep):
        lstLocs = []
        for dataLoc in lstLocData:
            locsN = []
            nK = len(dataLoc)
            iS = np.random.randint(0,nK,size=2)
            for j in iS:
                locsN.append(dataLoc[j])
            lstLocs.append(np.array(locsN))
            
        lstScore, lstSameLoc = runComparison(lstLocs)
        idxSameLoc = np.where(np.array(lstSameLoc) == 1)[0]
        idxDiffLoc = np.where(np.array(lstSameLoc) == 0)[0]
        clf = dictData['LOGISTIC_REG']
        thr = dictData['THR']
        LLRData = clf.predict_log_proba(np.array(lstScore).reshape(-1, 1))
        LLRRatio =  LLRData[:,1]-LLRData[:,0]
        same_LOC_data = LLRRatio[idxSameLoc]
        diff_LOC_data = LLRRatio[idxDiffLoc]
        P = len(same_LOC_data)
        N = len(diff_LOC_data)
        TP = np.sum(same_LOC_data > thr)
        TN = np.sum(diff_LOC_data <= thr)
        FN = np.sum(same_LOC_data <= thr)
        FP = np.sum(diff_LOC_data > thr)
        ACC.append(100*(TP+TN)/(P+N))
        FPR.append(100*FP/N)
        FNR.append(100*FN/P)
    return ACC, FPR, FNR
# =============================================================================

fpx = __file__.split("/")[-1].split("_")[0]
cm = 1/2.54  # centimeters in inches
DATA_FILE = '../PreProcessamento/csvLeeData_02_05_2024.csv'
df_VOGAIS = pd.read_csv(DATA_FILE,sep='\t')
lstLocutores = np.array(df_VOGAIS["Locutor"].unique())
lstFeatures = df_VOGAIS.columns[2:28]
lstFonetica = np.array(df_VOGAIS["Fonetica"].unique())
nLoc = len(lstLocutores)
nFeat = len(lstFeatures)
lstTagsFeat = ['Duração', 'média F01', 'CoV F01', 'média F02', 'CoV F02', 
               'média F03', 'CoV F03', 'média F04', 'CoV F04', 'média FD', 'CoV FD',
               'Intensidade', 'média F0', 'CoV F0', 'média SHR', 'CoV SHR', 
               'média CPPm', 'CoV CPP', 'média H1*-H2*', 'CoV H1*-H2*', 
               'média H2*-H4*', 'CoV H2*-H4*', 'média H4*-H2kHz*', 'CoV H4*-H2kHz*',
               'média H2kHz*-H5kHz*', 'CoV H2kHz*-H5kHz*']

ind_Variables = ['Sexo','Ditongo','Tonicidade','Fechada','Silabas','Posiçao','Oral']
idxFeatArt = [i for i in range(1,11)]
idxFeatVoc = [i for i in range(12,26)]

lstVogais = []
lstLocSex = []
lstLocSexTag = []
for Loc in lstLocutores:
    df_VogLoc = df_VOGAIS[(df_VOGAIS["Locutor"] == Loc)]
    valSex = df_VogLoc.loc[df_VogLoc.first_valid_index(),'Sexo']
    lstLocSexTag.append(valSex.replace("'",""))
    lstLocSex.append(df_VogLoc.loc[df_VogLoc.first_valid_index(),'Sexo'] == "'F'")
    lstVogais.append(np.array(df_VogLoc['Fonetica'].unique()))

FEATURE_NORMALIZATION = True
if (FEATURE_NORMALIZATION):
    for kFeat in lstFeatures:
        df_VOGAIS.loc[df_VOGAIS.index,kFeat] = (df_VOGAIS.loc[df_VOGAIS.index,kFeat] - df_VOGAIS[kFeat].mean())/df_VOGAIS[kFeat].std()
# sys.exit()
VOWEL_STATISTICS =  False
if (VOWEL_STATISTICS):
    vowelStats = df_VOGAIS.groupby('Fonetica').count()['ID']    
    vCat = np.array([FoneticaToTable(vowel) for vowel in  np.array(vowelStats.index)])
    vCount = np.array(vowelStats)
    idxSort = np.flip(np.argsort(vCount))
    vSortCount = vCount[idxSort]
    vSortCat = vCat[idxSort]
    vPercCount = 100*vSortCount/np.sum(vSortCount)
    
    vCumPercCount = np.cumsum(vPercCount)
    
    # nCut = np.where(vCumPercCount > (vCumPercCount[-1] - 0.5*vCumPercCount[0]))[0][0] + 1
    nCut = np.where(vCumPercCount > (vCumPercCount[-1] - 1.5*vCumPercCount[0]))[0][0] + 1
    
    selCount = np.append(vSortCount[0:nCut],np.sum(vSortCount[nCut:]))
    selCat   = np.append(vSortCat[0:nCut],'outros')
    # selCat   = np.append(['/{:}/'.format(x) for x in vSortCat[0:nCut]],'outros')
    
    vBase = [i for i in range(0,len(selCat))]
    vPercSel = 100*selCount/np.sum(selCount)
    vCumPercSel = np.cumsum(vPercSel)
    
    lstVowelLoc = []
    lstVowelLocPerc = []
    for iLoc in lstLocutores:
        dfLoc = df_VOGAIS[(df_VOGAIS["Locutor"] == iLoc)]
        lstForVowel = []
        for iV, iVowl in enumerate(selCat):
            vecVowelCut = [FoneticaToTable(iVwl) for iVwl in np.array(dfLoc["Fonetica"])]
            if (iV < nCut):
                lstForVowel.append(np.sum([iVowl == i for i in vecVowelCut]))
            else:
                lstForVowel.append(len(vecVowelCut)- np.sum(lstForVowel))
        
        lstVowelLoc.append(lstForVowel)
        lstVowelLocPerc.append(100*(lstForVowel/np.sum(lstForVowel)))
    
    dictForVowel = {}
    for iV, iVowl in enumerate(selCat):
        vecPerLocs = []
        for iLoc in range(0,len(lstLocutores)):
            vecPerLocs.append(lstVowelLoc[iLoc][iV])
        dictForVowel[iVowl] = vecPerLocs
    
    fig = plt.figure(figsize=(16*cm, 9*cm))
    plt.bar(vBase,vPercSel,label="Por Vogal")
    plt.plot(vBase,vCumPercSel,'ro-.',linewidth=0.5,label='Acumulado')
    plt.ylabel('Percentual')
    plt.xlabel('Vogais')
    plt.xticks(vBase,selCat)
    plt.legend()
    plt.grid(color="grey",linewidth=0.5, linestyle='-.')
    plt.savefig('Vowels_Count.png',bbox_inches="tight",dpi=200)
    
    buildTable(dictForVowel,lstLocutores,"CSV_01_Sons_Contexto.csv")
       
    width = 0.75
    fig, ax = plt.subplots(1,1, figsize =(9*cm, 16*cm))
    leftVal = np.zeros((len(lstLocutores),))
    for boolean, weight_count in dictForVowel.items():
        p = ax.barh(range(0,len(lstLocutores)), weight_count, width, label=boolean, left=leftVal)
        leftVal += weight_count
        
    ax.yaxis.set_ticks(range(0,len(lstLocutores)))
    ax.yaxis.set_ticklabels(lstLocutores)
    ax.set_title("Número de ocorrência das vogais por locutor locutores")
    ax.set_ylabel("Índice dos locutores")
    ax.set_xlabel("Número de ocorrências")
    ax.legend(ncol=1,loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim([-0.5,len(lstLocutores)-0.5])
    plt.gca().invert_yaxis()
    plt.grid(color='gray', linestyle='-.', linewidth=0.5)
    plt.savefig('PLT_01_Sons_Contexto.png',dpi=200,bbox_inches='tight') 
    
    buildTable(dictForVowel,lstLocutores,"CSV_01_Sons_Contexto.csv")
# =============================================================================


# --- Covariancia entre os locutores
mtxMeanFeatures = np.zeros((nFeat,nLoc))

commonFonetica = []
for kFon in lstFonetica:
    bInsert = True
    for setLoc in lstVogais:
        bInsert = bInsert and (kFon in setLoc)
        if not bInsert:
            break
    if (bInsert and (kFon not in commonFonetica)):
        commonFonetica.append(kFon) 


selShareTagFonet = [ajustVowelTag(X.replace("'","")) for X in commonFonetica if (X != "'ai'")]
selShareFonetica = [X for X in commonFonetica if (X != "'ai'")]

for i, Loc in enumerate(lstLocutores):
    df_VogLoc = df_VOGAIS[(df_VOGAIS["Locutor"] == Loc) & (df_VOGAIS['Fonetica'].isin(commonFonetica))]
    for j, Feat in enumerate(lstFeatures):
        mtxMeanFeatures[j,i] = np.mean(df_VogLoc[Feat])


COMPUTE_PCA = True
PLOT_CORR_PCA = False
PCA_FILE = '{:}_dict_PCA_File.pth'.format(fpx)
if (COMPUTE_PCA):
    dictPCA = {}
    corrFeatures = np.corrcoef(mtxMeanFeatures)
    w, vr = linalg.eig(corrFeatures,right=True)
    eigVals = np.abs(w)
    eigVals = eigVals/eigVals[0]
    csEigVals = np.cumsum(eigVals)
    csEigVals = csEigVals/csEigVals[-1]
    eigBase = range(0,nFeat)
    nPC = np.where(csEigVals > 0.95)[0][0]
    pca = PCA(n_components=nPC)
    B2 = pca.fit_transform(mtxMeanFeatures.T, y=lstTagsFeat)
    dictPCA["PCA"] = pca
    with open(PCA_FILE, "wb") as dill_file:
        dill.dump(dictPCA, dill_file)    
else:
    with open(PCA_FILE, "rb") as dill_file:
        dictPCA = dill.load(dill_file)  
    pca = dictPCA["PCA"]
    nPC = pca.n_components_
    eigBase = range(0,nFeat)
    
    
if (PLOT_CORR_PCA):    
    fig = plt.figure(figsize=(16*cm, 9*cm))
    plt.bar(eigBase,eigVals,label="Autovalores")
    plt.plot(eigBase,csEigVals,'r-.',linewidth=2,label='Soma dos autovalores')
    plt.plot([nPC-1, nPC-1],[0,1],'k--',linewidth=1)
    plt.plot([0, nFeat-1],[0.95,0.95],'k--',linewidth=1)
    plt.plot([nPC-1],[0.95],'ko',linewidth=1)
    plt.ylabel('Autovalores normalizados')
    plt.xlabel('ìndice do autovalor')
    plt.legend()
    plt.grid(color="grey",linewidth=0.5, linestyle='-.')
    plt.savefig('EiVals_features.png',bbox_inches="tight",dpi=300)
    
    fig = plt.figure(figsize=(20*cm, 20*cm))
    plt.imshow(corrFeatures)
    plt.clim(-1, 1)
    plt.grid(False)
    plt.xticks(range(0,nFeat), labels=lstTagsFeat, rotation=90)
    plt.yticks(range(0,nFeat), labels=lstTagsFeat)
    plt.title('Mapa de correlação entre as variáveis acústicas', fontsize=12)
    for i in range(nFeat):
         for j in range(nFeat):
             plt.text(j, i, "{:.2f}".format(corrFeatures[i, j]), ha='center', va='center', color='r', fontsize=6)
    cax = plt.axes((0.925, 0.125, 0.05, 0.755))
    plt.colorbar(cax=cax, format='% .2f')
    
    plt.savefig('Correelacao_features.png',bbox_inches="tight",dpi=300)
    
    corrPCA = np.corrcoef(B2.T)
    
    tagPrCmp = ['PCA {:02d}'.format(x) for x in range(0,nPC)]
    
    fig = plt.figure(figsize=(10*cm, 10*cm))
    plt.imshow(corrPCA)
    plt.clim(-1, 1)
    plt.grid(False)
    plt.xticks(range(0,nPC), labels=tagPrCmp, rotation=90)
    plt.yticks(range(0,nPC), labels=tagPrCmp)
    plt.title('Mapa de correlação entre as componentes principais', fontsize=12)
    for i in range(nPC):
         for j in range(nPC):
             plt.text(j, i, "{:.2f}".format(corrPCA[i, j]), ha='center', va='center', color='r', fontsize=6)
    cax = plt.axes((0.925, 0.125, 0.05, 0.755))
    plt.colorbar(cax=cax, format='% .2f')
    plt.savefig('Correelacao_pca.png',bbox_inches="tight",dpi=300)

    corrPCALoc = np.corrcoef(B2)
    ref = corrPCALoc[:,0]
    decIDX = np.flip(np.argsort(ref))
    corrPCALoc = np.corrcoef(B2[decIDX,:])

    # a = r'^{:}'.format()
    # 'Prior $%s$'%a.forma
    tagPrCmp = ['Locutor {:02d}'.format(lstLocutores[x]) for x in decIDX]
    tagPrCmpS = []
    for x in decIDX:
        a = r'^{:}'.format(lstLocSexTag[x])
        b = 'Locutor$%s$'%a
        tagPrCmpS.append('{:} {:02d}'.format(b,lstLocutores[x]))
        
    fig = plt.figure(figsize=(16*cm, 16*cm))
    plt.imshow(corrPCALoc)
    plt.clim(-1, 1)
    plt.grid(False)
    plt.xticks(range(0,nLoc), labels=tagPrCmpS, rotation=90)
    plt.yticks(range(0,nLoc), labels=tagPrCmpS)
    plt.title('Mapa de correlação entre os locutoress no espaço das componentes principais', fontsize=12)
    for i in range(nLoc):
         for j in range(nLoc):
             plt.text(j, i, "{:.2f}".format(corrPCALoc[i, j]), ha='center', va='center', color='r', fontsize=6)
    cax = plt.axes((0.925, 0.125, 0.05, 0.755))
    plt.colorbar(cax=cax, format='% .2f')
    plt.savefig('Correelacao_pca_loc.png',bbox_inches="tight",dpi=300)
    
    corrLoc = np.corrcoef(mtxMeanFeatures.T)

    fig = plt.figure(figsize=(16*cm, 16*cm))
    plt.imshow(corrLoc)
    # plt.clim(-1, 1)
    plt.grid(False)
    plt.xticks(range(0,nLoc), labels=tagPrCmp, rotation=90)
    plt.yticks(range(0,nLoc), labels=tagPrCmp)
    plt.title('Mapa de correlação entre os locutoress no espaço das variáveis acústicas', fontsize=12)
    for i in range(nLoc):
         for j in range(nLoc):
             plt.text(j, i, "{:.2f}".format(corrLoc[i, j]), ha='center', va='center', color='r', fontsize=6)
    cax = plt.axes((0.925, 0.125, 0.05, 0.755))
    plt.colorbar(cax=cax, format='% .2f')
    plt.savefig('Correelacao_loc.png',bbox_inches="tight",dpi=300)
# =============================================================================



# ----------------------------------------------------------------------------- 
dictFeaturePhonUnit = {}
for k, phoUnit in enumerate(selShareFonetica):
    dictFeat = {}
    for feat in lstFeatures:
        dictLoc = {}
        for i, iLoc in enumerate(lstLocutores):
            vecLocI = np.array(df_VOGAIS[(df_VOGAIS["Locutor"] == iLoc) & (df_VOGAIS['Fonetica'] == phoUnit)][feat])
            dictLoc[i] = vecLocI
        dictFeat[feat] = dictLoc
    dictFeaturePhonUnit[selShareTagFonet[k]] = dictFeat

shareVogaisOrais = np.array(selShareFonetica)[[0,1,2,3,4,8,10]]
shareVogaisOraisTag = np.array(selShareTagFonet)[[0,1,2,3,4,8,10]]

df_SharePhono = df_VOGAIS[(df_VOGAIS['Fonetica'].isin(commonFonetica))]

PHON_CNG_DATA_FILE = '{:}_PHCG.{:}'.format(*DATA_FILE.split("/")[-1].split("."))
CHANGE_PHONOLOGICAL_VARIABLES = True


VOWEL_AS_FEATURE = False
# colName = ['CLASS_{:02}'.format(i) for i in range(0,14)]
# colName = ['CLASS_{:02}'.format(i) for i in range(0,10)]
if (VOWEL_AS_FEATURE):
    colName = ['CLASS_{:02}'.format(i) for i in range(0,15)]
else:
    colName = ['CLASS_{:02}'.format(i) for i in range(0,10)]
if (CHANGE_PHONOLOGICAL_VARIABLES):
    df_factors = df_VOGAIS.columns
    featList = []
    idx = [i for i in range(0,df_SharePhono.shape[0])]
    for idx in range(0,df_VOGAIS.shape[0]):
        line = np.array(df_VOGAIS.loc[idx,["Precedente","Seguinte","Fonetica"]])
        # newFeat = consonantBinaryClass(line[0]) + vowelBinaryClass(line[0]) + consonantBinaryClass(line[1]) + vowelBinaryClass(line[1]) # + [len(grafPalavra)]
        if (VOWEL_AS_FEATURE):
            newFeat = soundBinaryClass(line[0]) + soundBinaryClass(line[1]) + soundBinaryClass(line[2])
        else:
            newFeat = soundBinaryClass(line[0]) + soundBinaryClass(line[1])
        featList.append(newFeat)
    df = pd.DataFrame(featList, columns=colName)
    df_PHVOGAIS = pd.concat([df_VOGAIS, df], axis=1)
    df_PHVOGAIS.to_csv(PHON_CNG_DATA_FILE,sep='\t',encoding='utf-8')
else:
    df_PHVOGAIS = pd.read_csv(PHON_CNG_DATA_FILE,sep='\t')

# sys.exit(" --- Depura 03 ----")

REBUILD_MODEL = False
RESAMPLE_DATA = False
MODEL_FILE = '{:}_dict_Modelo_File.pth'.format(fpx)
ind_Variables = [*ind_Variables, *colName]

SAMPLE_FILE = '{:}_dict_Sample_File.pth'.format(fpx)
idxTrain, idxTest = sampleTrainTestFromDataFrame(df_PHVOGAIS,SAMPLE_FILE, ind_Variables, RESAMPLE_DATA)

strIndVar = "C({:})".format(ind_Variables[0])
for i in range(1,len(ind_Variables)):
    strIndVar = strIndVar + " + C({:})".format(ind_Variables[i])
# strIndVar = strIndVar + "+ 1"    
if ((not os.path.exists(MODEL_FILE)) or REBUILD_MODEL):      
    dictModels = {}
    df_TrainSet = df_PHVOGAIS.iloc[idxTrain,:]
    ssVar = StandardScaler()
    for variable in lstFeatures:
        ssVar.fit(np.array(df_TrainSet[variable]).reshape(-1, 1))
        df_TrainSet.loc[df_TrainSet.index,variable] = ssVar.transform(np.array(df_TrainSet[variable]).reshape(-1, 1))
        md = smf.glm("{:} ~ {:}".format(variable,strIndVar), df_TrainSet)
        # md = smf.rlm("{:} ~ {:}".format(variable,strIndVar), df_TrainSet)
        mdf = md.fit()
        dictModels[variable] = {"Modelo": mdf,
                                "Scaler": ssVar} 
    residTrainData = np.zeros((len(idxTrain),len(lstFeatures)))
    rawTrainData = np.zeros((len(idxTrain),len(lstFeatures)))
    for idx, variable in enumerate(lstFeatures):
        mdf = dictModels[variable]["Modelo"]
        resY = np.array(df_TrainSet[variable])
        npResid = np.array(resY - mdf.predict())
        residTrainData[:,idx] = npResid
        rawTrainData[:,idx] = df_TrainSet[variable]
        
    dictModels["dependent_variables"] = lstFeatures
    with open(MODEL_FILE, "wb") as dill_file:
        dill.dump(dictModels, dill_file)
else:
    with open(MODEL_FILE, "rb") as dill_file:
        dictModels = dill.load(dill_file)        
    lstFeatures = dictModels["dependent_variables"]

RESID_TRAIN_SPEAKERS = False
TRAIN_SMF_FILE = '{:}_dict_Train_SMF_File.pth'.format(fpx)
if (RESID_TRAIN_SPEAKERS):
    if ( (not os.path.exists(TRAIN_SMF_FILE)) or REBUILD_MODEL):
        df_TrainSet = df_PHVOGAIS.iloc[idxTrain,:]
        listLocResid = []
        for variable in lstFeatures:
            ssVar = dictModels[variable]["Scaler"]
            df_TrainSet.loc[df_TrainSet.index,variable] = ssVar.transform(np.array(df_TrainSet[variable]).reshape(-1, 1))
        for idxL, loc in enumerate(lstLocutores):
            idxLoc = (np.array(df_TrainSet["Locutor"]) == loc).nonzero()[0]
            dictResidByVar = {}
            for variable in lstFeatures:
                mdf = dictModels[variable]["Modelo"]
                resY = np.array(df_TrainSet[variable])
                npResid = np.array(resY - mdf.predict())
                dictResidByVar[variable] = npResid[idxLoc]     
            listLocResid.append(dictResidByVar)
           
            
        dictModels["trainSpeakesFullResid"] = listLocResid
        # dictModels["whiteMatrix"] = W
        with open(TRAIN_SMF_FILE, "wb") as dill_file:
            dill.dump(dictModels, dill_file)
    else:
        with open(TRAIN_SMF_FILE, "rb") as dill_file:
            dictModels = dill.load(dill_file)        
        listLocResid  = dictModels["trainSpeakesFullResid"]
else:
    with open(TRAIN_SMF_FILE, "rb") as dill_file:
        dictModels = dill.load(dill_file)        
    listLocResid  = dictModels["trainSpeakesFullResid"]
    

RUN_TEST = False
TEST_SMF_FILE = '{:}_dict_Test_SMF_File.pth'.format(fpx)
if (RUN_TEST):
    if ( (not os.path.exists(TEST_SMF_FILE)) or REBUILD_MODEL):
        df_TestSet = df_PHVOGAIS.iloc[idxTest,:]
        locModelsTs = np.zeros((nFeat,nLoc))
        locModelsRawTs = np.zeros((nFeat,nLoc))
        listLocResidTs = []
        for variable in lstFeatures:
            ssVar = dictModels[variable]["Scaler"]
            df_TestSet.loc[df_TestSet.index,variable] = ssVar.transform(np.array(df_TestSet[variable]).reshape(-1, 1))
            
        for idxL, loc in enumerate(lstLocutores):
            idxLoc = (np.array(df_TestSet["Locutor"]) == loc).nonzero()[0]
            locSet = df_TestSet.iloc[idxLoc,:]
            locX = locSet[ind_Variables]
            dictResidByVar = {}
            for variable in lstFeatures:
                mdf = dictModels[variable]["Modelo"]
                resY = locSet.loc[:,variable]
                predY = np.array(mdf.predict(locSet))
                npResid = np.array(resY - predY)
                dictResidByVar[variable] = npResid
                
            listLocResidTs.append(dictResidByVar)
            
        dictModels["testSpeakesFullResid"] = listLocResidTs
        with open(TEST_SMF_FILE, "wb") as dill_file:
            dill.dump(dictModels, dill_file)
    else:
        with open(TEST_SMF_FILE, "rb") as dill_file:
            dictModels = dill.load(dill_file)        
        # locModelsTs = dictModels["testSpeakes"]
        listLocResidTs = dictModels["testSpeakesFullResid"]
        # locModelsRawTs = dictModels["testSpeakesRaw"]
else:
     with open(TEST_SMF_FILE, "rb") as dill_file:
         dictModels = dill.load(dill_file)        
     # locModelsTs = dictModels["testSpeakes"] 
     listLocResidTs = dictModels["testSpeakesFullResid"]
     # locModelsRawTs = dictModels["testSpeakesRaw"]

RUN_SAMPLE_SPEAKES = False
BOOTSTRAP_SPK_SAMPLE_FILE = '{:}_dict_BootStrap_Speaker_Sample_File.pth'.format(fpx)
if (RUN_SAMPLE_SPEAKES):
    dictBootStrapSpkSample = {}
    lstSMFLocsTrain = []
    lstSMFLocsTest = []
    for i, iLoc in enumerate(lstLocutores):
        vecLocTr = []
        vecLocTs = []
        for k, feat in enumerate(lstFeatures):
            ssVar = dictModels[feat]["Scaler"]
            if ( k == 0):
                vecLocTr = listLocResid[i][feat]
                vecLocTs = listLocResidTs[i][feat]
            else:
                # vTmpFeat = ssVar.transform(listLocResid[i][feat].reshape(-1,1)).ravel()
                vTmpFeat = listLocResid[i][feat]
                vecLocTr = np.vstack((vecLocTr,vTmpFeat))
                # vTmpFeat = ssVar.transform(listLocResidTs[i][feat].reshape(-1,1)).ravel()
                vTmpFeat = listLocResidTs[i][feat]
                vecLocTs = np.vstack((vecLocTs,vTmpFeat))
        vecLocTrS, vecLocTsS = resampleTrainTestOFRandonMeans(vecLocTr.T,vecLocTs.T)
        lstSMFLocsTrain.append(vecLocTrS)
        lstSMFLocsTest.append(vecLocTsS)
    dictBootStrapSpkSample["SMF_Train"] = lstSMFLocsTrain
    dictBootStrapSpkSample["SMF_Test"]  = lstSMFLocsTest
    # -------------------------------------------------------------------------
    lstPCALocsTrain = []
    lstPCALocsTest = []
    lstRAWLocsTrain = []
    lstRAWLocsTest = []
    for i, iLoc in enumerate(lstLocutores):
        vecLocI = []
        for feat in lstFeatures:
            vecLocI.append(df_VOGAIS[(df_VOGAIS["Locutor"] == iLoc)][feat].values)
        C1 = pca.transform(np.array(vecLocI).T)
        mTrain, mTest = sampleTrainTestOFRandonMeans(C1)
        lstPCALocsTrain.append(mTrain)
        lstPCALocsTest.append(mTest)
        mTrain, mTest = sampleTrainTestOFRandonMeans(np.array(vecLocI).T)
        lstRAWLocsTrain.append(mTrain)
        lstRAWLocsTest.append(mTest) 
    dictBootStrapSpkSample["PCA_Train"] = lstPCALocsTrain
    dictBootStrapSpkSample["PCA_Test"]  = lstPCALocsTest
    dictBootStrapSpkSample["RAW_Train"] = lstRAWLocsTrain
    dictBootStrapSpkSample["RAW_Test"]  = lstRAWLocsTest
    with open(BOOTSTRAP_SPK_SAMPLE_FILE, "wb") as dill_file:
        dill.dump(dictBootStrapSpkSample, dill_file)
else:
    with open(BOOTSTRAP_SPK_SAMPLE_FILE, "rb") as dill_file:
        dictBootStrapSpkSample = dill.load(dill_file)  
    lstSMFLocsTrain = dictBootStrapSpkSample["SMF_Train"]
    lstSMFLocsTest = dictBootStrapSpkSample["SMF_Test"]
    lstPCALocsTrain = dictBootStrapSpkSample["PCA_Train"]
    lstPCALocsTest = dictBootStrapSpkSample["PCA_Test"]
    lstRAWLocsTrain =  dictBootStrapSpkSample["RAW_Train"] 
    lstRAWLocsTest = dictBootStrapSpkSample["RAW_Test"]
    

lstDispersion = []

lstTagsDisp = ['GLM-RES treinamento','GLM-RES teste','PCA treinamento','PCA  teste','VA treinamento','VA  teste']
lstDispersion.append(ratioDispersionSet(lstSMFLocsTrain))
lstDispersion.append(ratioDispersionSet(lstSMFLocsTest))
lstDispersion.append(ratioDispersionSet(lstPCALocsTrain))
lstDispersion.append(ratioDispersionSet(lstPCALocsTest))
lstDispersion.append(ratioDispersionSet(lstRAWLocsTrain))
lstDispersion.append(ratioDispersionSet(lstRAWLocsTest))
vecMeanDispersion = np.array([ m for (m,x) in lstDispersion])
vecStdDispersion = np.array([ x for (m,x) in lstDispersion])

fig = plt.figure(figsize=(12*cm, 12*cm))
plt.barh(range(0,6),vecMeanDispersion, color='b',alpha=0.5,label='média')
plt.errorbar(vecMeanDispersion,range(0,6),xerr=vecStdDispersion,
             linestyle=' ',marker='s', capsize=4, color='k',label='erro padrão')
plt.grid(color='gray', linestyle='-.', linewidth=0.5)
plt.yticks(range(0,6),labels=lstTagsDisp)
plt.ylim([-0.5,5.5])
plt.xlabel('Dispersão normalizada')
plt.legend()
plt.title('Dispersão no espaço das características', fontsize=12)
plt.savefig('Dispersao.png',dpi=200,bbox_inches='tight')
    
RUN_COMPARISON = True
if (RUN_COMPARISON):
    lstScoreLocSMFTrain, lstSameLocSMFTrain = runComparison(lstSMFLocsTrain)
    lstScoreLocSMFTest, lstSameLocSMFTest = runComparison(lstSMFLocsTest)
    lstScoreLocPCATrain, lstSameLocPCATrain = runComparison(lstPCALocsTrain)
    lstScoreLocPCATest, lstSameLocPCATest = runComparison(lstPCALocsTest)
    lstScoreLocRAWTrain, lstSameLocRAWTrain = runComparison(lstRAWLocsTrain)
    lstScoreLocRAWTest, lstSameLocRAWTest = runComparison(lstRAWLocsTest)
    
    dictLR_SMF = runLogisticRegression(lstScoreLocSMFTrain, lstSameLocSMFTrain,lstScoreLocSMFTest, lstSameLocSMFTest)
    dictLR_PCA = runLogisticRegression(lstScoreLocPCATrain, lstSameLocPCATrain,lstScoreLocPCATest, lstSameLocPCATest)
    dictLR_RAW = runLogisticRegression(lstScoreLocRAWTrain, lstSameLocRAWTrain,lstScoreLocRAWTest, lstSameLocRAWTest)
    
    lstScoreLocSMFTrainAr, lstSameLocSMFTrainAr = runComparison(np.array(lstSMFLocsTrain)[:,:,idxFeatArt])
    lstScoreLocSMFTestAr, lstSameLocSMFTestAr = runComparison(np.array(lstSMFLocsTest)[:,:,idxFeatArt])
    lstScoreLocSMFTrainVc, lstSameLocSMFTrainVc = runComparison(np.array(lstSMFLocsTrain)[:,:,idxFeatVoc])
    lstScoreLocSMFTestVc, lstSameLocSMFTestVc = runComparison(np.array(lstSMFLocsTest)[:,:,idxFeatVoc])
    
    dictLR_SMFAr = runLogisticRegression( lstScoreLocSMFTrainAr, lstSameLocSMFTrainAr,lstScoreLocSMFTestAr, lstSameLocSMFTestAr)
    dictLR_SMFVc = runLogisticRegression(lstScoreLocSMFTrainVc, lstSameLocSMFTrainVc,lstScoreLocSMFTestVc, lstSameLocSMFTestVc)
    
    dictLR_SMFAr["Score_Train"] = lstScoreLocSMFTrainAr
    dictLR_SMFAr["SameLoc_Train"] = lstSameLocSMFTrainAr
    dictLR_SMFAr["Score_Test"] = lstScoreLocSMFTestAr
    dictLR_SMFAr["SameLoc_Test"] = lstSameLocSMFTestAr
    
    
    dictLR_SMFVc["Score_Train"] = lstScoreLocSMFTrainVc
    dictLR_SMFVc["SameLoc_Train"] = lstSameLocSMFTrainVc
    dictLR_SMFVc["Score_Test"] = lstScoreLocSMFTestVc
    dictLR_SMFVc["SameLoc_Test"] = lstSameLocSMFTestVc
    
    dictLR_SMF["Score_Train"] = lstScoreLocSMFTrain
    dictLR_SMF["SameLoc_Train"] = lstSameLocSMFTrain
    dictLR_SMF["Score_Test"] = lstScoreLocSMFTest
    dictLR_SMF["SameLoc_Test"] = lstSameLocSMFTest
    
    dictLR_PCA["Score_Train"] = lstScoreLocPCATrain
    dictLR_PCA["SameLoc_Train"] = lstSameLocPCATrain
    dictLR_PCA["Score_Test"] = lstScoreLocPCATest
    dictLR_PCA["SameLoc_Test"] = lstSameLocPCATest
    
    dictLR_RAW["Score_Train"] = lstScoreLocRAWTrain
    dictLR_RAW["SameLoc_Train"] = lstSameLocRAWTrain
    dictLR_RAW["Score_Test"] = lstScoreLocRAWTest
    dictLR_RAW["SameLoc_Test"] = lstSameLocRAWTest
    
    with open("dict_SMF.pth", "wb") as dill_file:
        dill.dump(dictLR_SMF, dill_file)
    with open("dict_PCA.pth", "wb") as dill_file:
        dill.dump(dictLR_PCA, dill_file)
    with open("dict_RAW.pth", "wb") as dill_file:
        dill.dump(dictLR_RAW, dill_file)
    with open("dict_SMF_AR.pth", "wb") as dill_file:
        dill.dump(dictLR_SMFAr, dill_file)
    with open("dict_SMF_VC.pth", "wb") as dill_file:
        dill.dump(dictLR_SMFVc, dill_file)
else:
    with open("dict_SMF.pth", "rb") as dill_file:
        dictLR_SMF = dill.load(dill_file)  
    with open("dict_PCA.pth", "rb") as dill_file:
        dictLR_PCA = dill.load(dill_file)  
    with open("dict_RAW.pth", "rb") as dill_file:
        dictLR_RAW = dill.load(dill_file)  
    with open("dict_SMF_AR.pth", "rb") as dill_file:
        dictLR_SMFAr = dill.load(dill_file)  
    with open("dict_SMF_VC.pth", "rb") as dill_file:
        dictLR_SMFVc = dill.load(dill_file)  


smfACC, smfFPR, smfFNR =  leaveOneOutJack(dictLR_SMF,lstSMFLocsTrain, 20)
pcaACC, pcaFPR, pcaFNR =  leaveOneOutJack(dictLR_PCA,lstPCALocsTrain, 20)
rawACC, rawFPR, rawFNR =  leaveOneOutJack(dictLR_RAW,lstRAWLocsTrain, 20)
smfACCar, smfFPRar, smfFNRar =  leaveOneOutJack(dictLR_SMFAr,np.array(lstSMFLocsTrain)[:,:,idxFeatArt], 20)
smfACCvc, smfFPRvc, smfFNRvc =  leaveOneOutJack(dictLR_SMFVc,np.array(lstSMFLocsTrain)[:,:,idxFeatVoc], 20)



lstACC = [smfACC,pcaACC,rawACC,smfACCar,smfACCvc]
lstFPR = [smfFPR,pcaFPR,rawFPR,smfFPRar,smfFPRvc]
lstFNR = [smfFNR,pcaFNR,rawFNR,smfFNRar,smfFNRvc]
lstTags = ['GLM-RES','PCA','VA','GLM-RES-A','GLM-RES-V']

#t.interval(0.95, len(smfACC)-1, loc=np.mean(smfACC), scale=sem(smfACC))
for iV in lstACC:
    a,b = t.interval(0.95, len(iV)-1, loc=np.mean(iV), scale=sem(iV))
    print("{:.1f} ({:.1f}; {:.1f})".format(np.mean(iV),a,b).replace(".",","))
for iV in lstFPR:
    a,b = t.interval(0.95, len(iV)-1, loc=np.mean(iV), scale=sem(iV))
    print("{:.1f} ({:.1f}; {:.1f})".format(np.mean(iV),a,b).replace(".",","))
for iV in lstFNR:
    a,b = t.interval(0.95, len(iV)-1, loc=np.mean(iV), scale=sem(iV))
    print("{:.1f} ({:.1f}; {:.1f})".format(np.mean(iV),a,b).replace(".",","))    
    
tagsComp, limitsComp, _, _ = anovaResult(lstACC,lstTags,
                                            'Anova_Acuracia.txt')
meanPeca = [0.5*(limitsC[1] + limitsC[0]) for limitsC in limitsComp]
limitsAnova = np.array([ [m-lim[0],lim[1]-m] for lim, m in zip(limitsComp,meanPeca)] )
fig = plt.figure(figsize=(20*cm, 10*cm))
gs = fig.add_gridspec(nrows=1, ncols=3, wspace=0.05)
ax = gs.subplots(sharex=False,sharey=True)

ax[0].errorbar(meanPeca,range(0,len(meanPeca)),xerr=limitsAnova.T,
             linestyle=' ',marker='s', capsize=4, color='b')
ax[0].plot([0,0],[-0.25,len(meanPeca)-0.75], linestyle='-.',linewidth=2, color='r')
ax[0].set_yticks(range(0,len(meanPeca)),tagsComp)
ax[0].set_title("Acurácia")
ax[0].grid(color='gray', linestyle='-.', linewidth=0.5)
ax[0].set_xlabel('Diferença (%)')
ax[0].set_ylim([-0.25,len(meanPeca)-0.75])

tagsComp, limitsComp, _, _ = anovaResult(lstFPR,lstTags,
                                            'Anova_Falso_Positivo.txt')
meanPeca = [0.5*(limitsC[1] + limitsC[0]) for limitsC in limitsComp]
limitsAnova = np.array([ [m-lim[0],lim[1]-m] for lim, m in zip(limitsComp,meanPeca)] )

ax[1].errorbar(meanPeca,range(0,len(meanPeca)),xerr=limitsAnova.T,
             linestyle=' ',marker='s', capsize=4, color='b')
ax[1].plot([0,0],[-0.25,len(meanPeca)-0.75], linestyle='-.',linewidth=2, color='r')
ax[1].set_title("Falso positivo")
ax[1].grid(color='gray', linestyle='-.', linewidth=0.5)
ax[1].set_xlabel('Diferença (%)')
ax[1].set_ylim([-0.25,len(meanPeca)-0.75])

tagsComp, limitsComp, _, _ = anovaResult(lstFNR,lstTags,
                                            'Anova_Falso_Negativo.txt')
meanPeca = [0.5*(limitsC[1] + limitsC[0]) for limitsC in limitsComp]
limitsAnova = np.array([ [m-lim[0],lim[1]-m] for lim, m in zip(limitsComp,meanPeca)] )

ax[2].errorbar(meanPeca,range(0,len(meanPeca)),xerr=limitsAnova.T,
             linestyle=' ',marker='s', capsize=4, color='b')
ax[2].plot([0,0],[-0.25,len(meanPeca)-0.75], linestyle='-.',linewidth=2, color='r')
ax[2].set_title("Falso negativo")
ax[2].grid(color='gray', linestyle='-.', linewidth=0.5)
ax[2].set_xlabel('Diferença (%)')
ax[2].set_ylim([-0.25,len(meanPeca)-0.75])



plt.savefig('ANOVA_Acuracis_00.png',dpi=200,bbox_inches='tight')



plotLogisticResults(dictLR_SMF,'SMF')
plotLogisticResults(dictLR_PCA,'PCA')
plotLogisticResults(dictLR_RAW,'RAW')





a = r'C_{LLR}'
plot_file_name = "ECE_01.jpg"
fig = plt.figure(figsize =(12*cm, 12*cm))
plt.plot(np.log10(dictLR_SMF["ECE_x"]),dictLR_SMF["ECE_r"], color = 'gray', linewidth=2,linestyle='dotted')
plt.plot(np.log10(dictLR_SMF["ECE_x"]),dictLR_SMF["ECE_y"], color = 'blue', linewidth=2, 
          label="GLM-RES - ${:}$: {:4.3f}; EER: {:3.1f} %".format(a,dictLR_SMF["CLLR"],100*dictLR_SMF["EER"]).replace(".",","))
plt.plot(np.log10(dictLR_SMFAr["ECE_x"]),dictLR_SMFAr["ECE_y"], color = 'tab:olive', linewidth=2, 
          label="GLM-RES-ART - ${:}$: {:4.3f}; EER: {:3.1f} %".format(a,dictLR_SMFAr["CLLR"],100*dictLR_SMFAr["EER"]).replace(".",","))
plt.plot(np.log10(dictLR_SMFVc["ECE_x"]),dictLR_SMFVc["ECE_y"], color = 'tab:green', linewidth=2, 
          label="GLM-RES-VOC - ${:}$: {:4.3f}; EER: {:3.1f} %".format(a,dictLR_SMFVc["CLLR"],100*dictLR_SMFVc["EER"]).replace(".",","))
plt.plot(np.log10(dictLR_PCA["ECE_x"]),dictLR_PCA["ECE_y"], color = 'magenta', linewidth=2, 
          label="PCA - ${:}$: {:4.3f}; EER: {:3.1f} %".format(a,dictLR_PCA["CLLR"],100*dictLR_PCA["EER"]).replace(".",","))
plt.plot(np.log10(dictLR_RAW["ECE_x"]),dictLR_RAW["ECE_y"], color = 'orange', linewidth=2, 
          label="VA - ${:}$: {:4.3f}; EER: {:3.1f} %".format(a,dictLR_RAW["CLLR"],100*dictLR_RAW["EER"]).replace(".",","))
plt.plot([0,0],[0,1], color = 'black', linewidth=2,linestyle='dotted')
a = r'C_{LLR}'
# plt.title('Entropia cruzada empírica doconjunto de teste $%s$'%a)
plt.title('Entropia cruzada empítica do conjunto de teste')
plt.ylim(0,0.8)
plt.xlim(-2.5,2.5)
a = r'\log_{10}(chance)'
plt.xlabel('Probablidade a priori $%s$'%a)
plt.ylabel('Entropia cruzada empírica')
plt.grid()
plt.legend()
fig.savefig(plot_file_name,dpi=300,bbox_inches='tight')        
        

R2 = [dictModels[variable]["Modelo"].pseudo_rsquared() for variable in lstFeatures]
dictFeature = {}
listPval = []
listStats = []
for feat in lstFeatures:
    lstLoc = []
    for i, iLoc in enumerate(lstLocutores):
        vecLocI = np.array(df_VOGAIS[(df_VOGAIS["Locutor"] == iLoc)][feat])
        
        lstLoc.append(vecLocI)
    dictFeat[feat] = lstLoc
    anova = f_oneway(*lstLoc)
    # result = tukey_hsd(*lstLoc)
    # conf = result.confidence_interval(confidence_level=.95)
    listPval.append(anova.pvalue)
    listStats.append(anova.statistic)
dictFeature[feat] = dictFeat

ref = range(0,nFeat)

R2_ArtMean = [np.mean(np.array(R2)[idxFeatArt]) for i in idxFeatArt]
R2_VocMean = [np.mean(np.array(R2)[idxFeatVoc]) for i in idxFeatVoc]

fig = plt.figure(figsize=(18*cm, 12*cm))
gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0)
ax = gs.subplots(sharex=False,sharey=True)
a = r'R^{2}'
ax[0].barh(ref,listPval, color='g',alpha=0.5)
ax[0].plot([0.05,0.05],[-0.5,nFeat], linestyle='-.', color='k',linewidth=2)
# plt.clim(-1, 1)
ax[0].grid(color='gray', linestyle='-.', linewidth=0.5)
ax[0].set_yticks(ref,labels=lstTagsFeat)
ax[0].set_ylim([-0.5,nFeat])
ax[0].set_xlim([0,1.1])
ax[0].set_xlabel('valor-p')
ax[0].set_title('Valor-p da ANOVA entre as características', fontsize=12)

ax[1].barh(ref,R2, color='b',alpha=0.5)
ax[1].plot(R2_ArtMean,idxFeatArt, linestyle='-', color='tab:blue',linewidth=2)
ax[1].plot(R2_VocMean,idxFeatVoc, linestyle='-', color='tab:blue',linewidth=2)
# plt.clim(-1, 1)
ax[1].grid(color='gray', linestyle='-.', linewidth=0.5)
# ax[1].set_yticks(ref,labels=lstFeatures)
# ax[1].ylim([-0.5,nFeat])
ax[1].set_xlabel('pseudo $%s$'%a)
ax[1].set_title('pseudo $%s$ do GLM'%a, fontsize=12)

plt.savefig('pValue_ANOVA_features.png',bbox_inches="tight",dpi=300)

                   




sys.exit(' --- Parada ----')
