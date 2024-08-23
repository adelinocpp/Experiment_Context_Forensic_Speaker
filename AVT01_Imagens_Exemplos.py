#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cálculo de características acústicas em vogais
Minha humilde contribuição para o trabalho de Maria Cantoni e Thais Cristofaro

"""
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue
        del globals()[var]
        
clear_all()
import sys
sys.path.insert(1, '/home/adelino/MEGAsync/Academico/Avante_7a_Edicao/Modelos_Locutores_Desacoplamento_Contexto/ISSP_Modelo_Residuos/PreProcessamento')
from pputils.file_utils import list_contend, textgrid_to_interval_matrix #spectral_ratios
from scipy.io import wavfile
# from Signal_Analysis.features.signal import get_HNR #, get_Jitter, get_F_0
import numpy as np
from pathlib import Path

# import os
# import praat_formants_python as pfp
from pputils.formant_lpc import formant_lpc, intensity, formantDispersion
from g2p.g2p import G2PTranscriber
import re
import warnings
from scipy.signal import filtfilt, butter
# from scipy.signal.windows import gaussian
# from scipy.integrate import simpson
# from scipy import fft, stats
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
# from unidecode import unidecode
warnings.filterwarnings("ignore")
from scipy.signal import lfilter

from pputils.phono_utils import checkToncidade, is_ditongo, dinamicFormant
from pputils.phono_utils import getMeanPercentualInterval, get_H, estimate_syllabe_position
from pputils.phono_utils import  find_pos_of_tag, has_vogal, pos_vowel_in_word, pos_indicated_vowel

from pputils.cepstral_peak_proeminence import cpp
from pputils.shrp import shrp

    

AUDIO_FOLDER = '../Audios/'
audiofiles = list_contend(folder=AUDIO_FOLDER, pattern=('.wav',))
textgridfiles = list_contend(folder=AUDIO_FOLDER, pattern=('.textgrid',))

if (len(audiofiles) != len(textgridfiles)):
    print("Erro: número de arquivos de áudio não corresponde ao numero de TextGrid")

# Tamanho do passo de tempo em segundos
valStep = 0.005
valWin  = 0.020
nForm = 4

listConsoante = ('b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z')
list_phon_vowels = ("a", "e", "o", "á", "é", "í", "ó", "ú", "ã", "õ", "â", "ê", "ô", "à", "ü","ɪ", "ɛ", "ɐ", "ʊ", "ɔ")
list_phon_consonant = ('b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z','ʃ', 'ʎ', 'ɲ', 'ɳ', 'ɾ', 'ɣ', 'ʒ', 'ʤ', 'ʧ', 'ŋ')

fnTitle = ""
for i in range(1,nForm+1):
    fnTitle = fnTitle + "F{:02}m,F{:02}v,".format(i,i)
fnTitle = fnTitle + "Fdm,Fdv,"
f0Title = ""
f0Title = f0Title + "F0m,F0v,"
f0Title = f0Title + "SHRm,SHRv,"
f0Title = f0Title + "CPPm,CPPv,"
f0Title = f0Title + "H12m,H12v,"
f0Title = f0Title + "H24m,H24v,"
f0Title = f0Title + "H42km,H42kv,"
f0Title = f0Title + "H2k5km,H2k5kv,"
    
strTitle = "ID,Locutor,Duração,{:}intensidade,{:}tag,Posiçao,Fonetica,Fonologica,Ditongo,Palavra,Tonicidade,Precedente,Seguinte,Fechada,Silabas,Oral,LetraPre,LetraSeg,Sexo,Arquivo\n".format(fnTitle,f0Title).replace(",","\t")
csvLines = []
csvLines.append(strTitle)
useTiers = (0,)
tabId = 0;
maxNSyllab = 0


for idxtg, tgFile in enumerate(textgridfiles):
    value = textgrid_to_interval_matrix(tgFile)
    sr, audio = wavfile.read(audiofiles[idxtg])
    b, a = butter(4, [20/(0.5*sr), (0.5*sr-20)/(0.5*sr)] , btype='bandpass')    
    audio = filtfilt(b, a, audio)
    audioPreEmph = lfilter([1., -.975], 1, audio) 
    tabSexo = tgFile[-15]
    tabFileName = tgFile.split("/")[-1].split(".")[0]
    if (len(audio.shape) > 1):
        nSamples, nChannel = audio.shape
    else:
        nSamples  = len(audio)
        nChannel  = 1
    if (nChannel > 1):
        audio = np.mean(audio,axis = 1)
    audio = audio/np.max(np.abs(audio))
        
    for j in useTiers:
        intervalMtx = value[j]    
        for idxL, interval in enumerate(intervalMtx):
            if (tabId > 0):
                sys.exit(" --- Processado: tabId = {:}.".format(tabId))
            tabDitongo = 0
            nIni = int(interval[0]*sr)
            nFim = int(interval[1]*sr)
            tabDuration = interval[1] - interval[0]
            if (tabDuration < (valWin + valStep)):
                print("1: Segmento ID {:} de {:} seg é muito pequeno para processamento.".format(tabId,tabDuration))
                print("1: Arquivo {:} no intervalo entre {:2.3f} e {:2.3f} segundos.".format(Path(tgFile).name,interval[0],interval[1]))
                continue
            

            tags = interval[2].split("-")
            # --- Trata digito errado
            if (tags[0].isdigit()):
                if (tags[0] == '1'):
                    tags[0] = 'i'
                elif (tags[0] == '3'):
                    tags[0] = 'e'
                elif (tags[0] == '4'):
                    tags[0] = 'a'
                else:
                    tags[0] = 'o'
            
            # ----------------------
            tags[0] = tags[0].replace(" ","")
            
            if not (len(tags) == 5):
                print("2: Etiqueta {:} ID {:} com problema no número de marcações.".format(interval[2],tabId))
                print("2: Arquivo {:} no intervalo entre {:2.3f} e {:2.3f} segundos.".format(Path(tgFile).name,interval[0],interval[1]))
                continue
            if (len(tags[2]) > 0):
                try:
                    if (tags[4] == ''):
                        # tratavel
                        print("12: Problema TAG 4 {: em {:}}".format(tags[4],interval[2]))
                        continue
                    nSib = int(tags[2])
                    if (nSib > maxNSyllab):
                        maxNSyllab = nSib
                except:
                    print("9: Problema com a TAG 2 como {:} de ID {:} ".format(tags[2],tabId))
                    continue
            else:
                continue
            
            # --- Trata tag errada
            if not (tags[4].isdigit()):
                find = re.compile('(\d{1})')
                if (tags[4] == 'O') or (tags[4] == 'o'):
                    tags[4] =   '0'
                else:
                   # captura primeiro numero na string 
                   fstDig = re.match(find,tags[4])
                   tags[4] = "{:}".format(fstDig.start())
                       
            # ----------------------
            tabFonetica = tags[0]
            if (len(tabFonetica) == 1) and (tabFonetica.lower() in listConsoante):
                print("2: Etiqueta {:} ID {:} apresenta tag que nao e vogal {:}.".format(interval[2],tabId,tabFonetica))
                continue
            
            tabPalavra = tags[1]
            tabTonicidade = checkToncidade(tags[3])
            tabDitongo = is_ditongo(tags[0])
            
            if (tabDitongo):
                print("Salto de um ditongo")
                continue
                    
            
            selAudio = audio[nIni:nFim]
            selAudioPE = audioPreEmph[nIni:nFim]
            nFrames = int(np.floor(((len(selAudioPE)/sr) - valWin)/valStep + 1))
            cpp_Feat, cpp_t = cpp(selAudio,sr,'line',True, frame_step=valStep, frame_win=valWin)
            _, _, SHR, _ = shrp(selAudio,sr, frame_length=20, timestep=5)
            inten = intensity(selAudio,sr,winstep=valStep,winlen=valWin)
            vTimeBase = cpp_t + interval[0]
            try:
                # tabJitter = get_Jitter(selAudio,sr)
                # median_F_0, periods, _, _ = get_F_0(selAudio,sr,time_step=valStep,min_pitch=50,pulse = True )
                # pitch = 1.0/np.array(periods)
                yAudio = basic.SignalObj(selAudio,sr)
                pitch = pYAAPT.yaapt(yAudio,**{'f0_min' : 50.0, 'frame_length' : 1000*valWin, 'frame_space' : 1000*valStep})
                # pitch = pitch.samp_interp
            except:
                print("14: Nao foi possivel calcular F0 no arquivo {:} no intervalo entre {:2.3f} e {:2.3f} segundos.".format(Path(tgFile).name,interval[0],interval[1]))
                continue
            # tabSpecRatios =  spectral_ratios(selAudio,sr,time_step=valStep)
            
            if (np.prod(pitch.samp_values) < 0.01):
                print("Problema no calculo pitch")
                continue
            
            form2, band2 = formant_lpc(selAudio,sr, pitch.samp_values, nFormReq=5, maxFreq = 1.1*5000, winstep=valStep,winlen=valWin)
            form2 = form2[0:4,:]
            band2 = band2[0:4,:]
            
            # return H1, H2, H4, H2k e H5K
            HValues = get_H(selAudio,sr,pitch.samp_values, form2, band2,maxFreq = sr/2,winlen=valWin, winstep=valStep)
            H12 = [X[0] - X[1] for X in HValues]
            H24 = [X[1] - X[2] for X in HValues]
            H42k = [X[2] - X[3] for X in HValues]
            H2k5k = [X[3] - X[4] for X in HValues]
            
            
            tabIntensity = getMeanPercentualInterval(inten,0.2,0.8)
            
            
            tabF0 = []
            fZero, _ , FTwo = dinamicFormant(pitch.samp_interp,valStep)
            tabF0 = tabF0 + [fZero,  FTwo]
            fZero, _ , FTwo = dinamicFormant(SHR,valStep)
            tabF0 = tabF0 + [fZero,  FTwo]
            fZero, _ , FTwo = dinamicFormant(cpp_Feat,valStep)
            tabF0 = tabF0 + [fZero,  FTwo]
            
            
            fZero, _ , FTwo = dinamicFormant(H12,valStep)
            tabF0 = tabF0 + [fZero,  FTwo]
            fZero, _ , FTwo = dinamicFormant(H24,valStep)
            tabF0 = tabF0 + [fZero,  FTwo]
            fZero, _ , FTwo = dinamicFormant(H42k,valStep)
            tabF0 = tabF0 + [fZero,  FTwo]
            fZero, _ , FTwo = dinamicFormant(H2k5k,valStep)
            tabF0 = tabF0 + [fZero,  FTwo]
            
            tabFn = []
            for nF in range(0,nForm):
                fZero, _ , FTwo = dinamicFormant(form2[nF,:],valStep)
                tabFn = tabFn + [fZero, FTwo]
            fDisp = formantDispersion(form2)    
            fZero, _ , FTwo = dinamicFormant(fDisp,valStep)
            tabFn = tabFn + [fZero, FTwo]    
            
            
            
            
            # momento da transcricao groafica para fonetica
            g2p = G2PTranscriber(tabPalavra, algorithm='ceci')
            phonPalavra = g2p.transcriber().split(",")[0].split('.')
            grafPalavra = g2p.get_syllables_with_hyphen().split('-')
            
            if (len(grafPalavra) != len(phonPalavra)):
                phonPalavra = phonPalavra[:(len(grafPalavra)-1)]
            
            
            if (len(grafPalavra) != int(tags[2])):
                # Verifica se silabificaçao esta coerente
                silabOk = (len(grafPalavra) == len(phonPalavra))
                if silabOk and (len(grafPalavra) > int(tags[2])):
                    print("4: Arquivo {:} Numero de silabas alteradas de {:} para {:}.".format(Path(tgFile).name,tags[2],len(grafPalavra)))
                    tags[2] = "{:}".format(len(grafPalavra))
                else:
                    print('4: ID {:}. Problema na silabificação (tag {:}) de \"{:}\" como {:} , {:}'.format(tabId,int(tags[2]),tabPalavra,phonPalavra, grafPalavra))
                    print("4: Arquivo {:} no intervalo entre {:2.3f} e {:2.3f} segundos.".format(Path(tgFile).name,interval[0],interval[1]))
                    continue
                
            vogalPos = estimate_syllabe_position(phonPalavra,int(tags[2]),int(tags[4]))
            if (vogalPos < 0):
                
                print("5: Etiqueta {:} ID {:} com problema. Falha na estimaçao da posiçao da silaba".format(interval[2],tabId))
                print("5: Arquivo {:} no intervalo entre {:2.3f} e {:2.3f} segundos.".format(Path(tgFile).name,interval[0],interval[1]))
                continue

            try:
                sibPosition = int(tags[4])
                if (int(tags[4]) >  len(grafPalavra)):
                    # Tratavel ams tem muitas opçoes
                    print("6: Etiqueta {:} ID {:} com problema. Possicao maior que o numero de silabas".format(interval[2],tabId))
                    print("6: Arquivo {:} no intervalo entre {:2.3f} e {:2.3f} segundos.".format(Path(tgFile).name,interval[0],interval[1]))
                    continue
            except:
                print("10: Problema com a TAG 4 como {:} de ID {:} ".format(tags[4],tabId))
                continue
            
            
            # remove o marcador de silaba tonica
            phonSilaba = phonPalavra[vogalPos].replace("ˈ","")
            grapSilaba = grafPalavra[vogalPos]
            tabFonologico = phonSilaba #phonPalavra[vogalPos]
            #phonTrancrib = ''.join(phonPalavra).replace("ˈ","")
            # TODO:
                # Nao identifica algumas vogais tipo em " ʧĩ"
            
            hasNasal = False
            idxNasal = []
            if ('͂' in phonSilaba) or ('̃' in phonSilaba):
                hasNasal = True    
                oriphonSilaba = phonSilaba
                phonSilaba = phonSilaba.replace('͂','').replace('̃','')
                print("Tratar as nasais")
                
            
            if (not has_vogal(tabFonologico)):
                print("7: ID {:} com problema, {:} sem vogal.".format(tabId, tabFonologico))
                continue
            
            tabPrecedente = 'NA'
            tabSeguinte = 'NA'
            if (tabDitongo):
                vogLen = len(tags[0])
                pos = grapSilaba.find(tags[0])
                if (pos == -1):
                    pos, valT = find_pos_of_tag(grapSilaba,tags[0])
                elif (pos == -1):
                    print("13: Problema na detecçao da tag ID {:}".format(tabId))
                    continue
                if (pos == 0) and (len(phonSilaba) == vogLen):
                    tabPrecedente = 'NA'
                    tabSeguinte = 'NA'
                if (pos == 0) and (len(phonSilaba) > vogLen):
                    tabPrecedente = 'NA'
                    tabSeguinte = phonSilaba[1]
                if (pos > 0) and (len(phonSilaba) == (pos+vogLen)):
                    tabPrecedente = phonSilaba[pos-1]
                    tabSeguinte = 'NA'
                if (pos > 0) and (len(phonSilaba) > (pos+vogLen)):
                    tabPrecedente = phonSilaba[pos-1]
                    tabSeguinte = phonSilaba[pos+1]
            else:
                if (hasNasal):
                    print('Depurando...')
                pos = pos_indicated_vowel(phonSilaba, tags[0], tabId)
                if (pos == -1):
                    pos, valT = find_pos_of_tag(grapSilaba,tags[0])
                elif (pos == -1):
                    print("13: Problema na detecçao da tag ID {:}".format(tabId))
                    continue
                # pos = grapSilaba.find(tags[0])
                if (pos == 0) and (len(phonSilaba) == 1):
                    tabPrecedente = 'NA'
                    tabSeguinte = 'NA'
                if (pos == 0) and (len(phonSilaba) > 1):
                    tabPrecedente = 'NA'
                    tabSeguinte = phonSilaba[1]
                if (pos > 0) and (len(phonSilaba) == (pos+1)):
                    tabPrecedente = phonSilaba[pos-1]
                    tabSeguinte = 'NA'
                if (pos > 0) and (len(phonSilaba) > (pos+1)):
                    tabPrecedente = phonSilaba[pos-1]
                    tabSeguinte = phonSilaba[pos+1]
            
            
            tabLetraPre = 'NA'
            tabLetraSeg = 'NA'
            # if (tabId == 10):
            #     print('Depurando...')
            if (tabDitongo):
                vogLen = len(tags[0])
                pos = tabPalavra.find(tags[0])
                if (pos == -1):
                    pos, valT = find_pos_of_tag(tabPalavra,tags[0])
                elif (pos == -1):
                    print("13: Problema na detecçao da tag ID {:}".format(tabId))
                    continue
                if (pos == 0) and (len(tabPalavra) == vogLen):
                    tabLetraPre = 'NA'
                    tabLetraSeg = 'NA'
                if (pos == 0) and (len(tabPalavra) > vogLen):
                    tabLetraPre = 'NA'
                    tabLetraSeg = tabPalavra[pos+vogLen]
                if (pos > 0) and (len(tabPalavra) == (pos+vogLen)):
                    tabLetraPre = tabPalavra[pos-1]
                    tabLetraSeg = 'NA'
                if (pos > 0) and (len(tabPalavra) > (pos+vogLen)):
                    tabLetraPre = tabPalavra[pos-1]
                    tabLetraSeg = tabPalavra[pos+vogLen]
            else:
                pos = pos_vowel_in_word(grafPalavra, vogalPos, tags[0])
                if (pos == 0) and (len(grafPalavra) == 1):
                    tabLetraPre = 'NA'
                    tabLetraSeg = 'NA'
                if (pos == 0) and (len(tabPalavra) > 1):
                    tabLetraPre = 'NA'
                    tabLetraSeg = tabPalavra[pos+1]
                if (pos > 0) and (len(tabPalavra) == (pos+1)):
                    tabLetraPre = tabPalavra[pos-1]
                    tabLetraSeg = 'NA'
                if (pos > 0) and (len(tabPalavra) > (pos+1)):
                    tabLetraPre = tabPalavra[pos-1]
                    tabLetraSeg = tabPalavra[pos+1]
                    
            if (hasNasal):
                print("recolocar a nasal")
                phonSilaba = oriphonSilaba
                continue
            
            if ((tabLetraPre == 'NA') or (tabLetraSeg == 'NA')):
                print("Escolher palavra com fronteira")
                continue
            
            
            
            
            
            
            
            
            nIniPlt = int((interval[0] - 0.0)*sr)
            nFimPlt = int((interval[1] - 0.0)*sr)
            plotAudio = audio[nIniPlt:nFimPlt] 
            
            
            
            
            
            
            sys.exit(" --- Consegiu calcular -----")
            # TODO: Retirar redundancia de tabOral e hasNasal
            tabOral = int((not hasNasal))
            tabFechada = int((grapSilaba[-1] in listConsoante))
            
            
            
            tabData = (tabId,int(tabFileName.split("_")[1][:-1]), tabDuration,tabFn ,tabIntensity,
                       tabF0,interval[2],vogalPos,
                       tabFonetica,tabFonologico, tabDitongo, 
                       tabPalavra, tabTonicidade, tabPrecedente, tabSeguinte,tabFechada,
                       nSib, tabOral, tabLetraPre.lower(), tabLetraSeg.lower(),
                       tabSexo,tabFileName)
            
            strData = "{:}\n".format(tabData).replace("[","").replace("]","").replace("(","").replace(")","").replace(" ","").replace(",","\t")
            # sys.exit("Saida de depuraçao")
            csvLines.append(strData)
            tabId = tabId + 1
        # sys.exit("Saida de depuraçao - RODOU APENAS CAMADA 1!")
        
