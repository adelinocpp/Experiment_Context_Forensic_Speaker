#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 07:53:40 2024

@author: adelino
"""

# -----------------------------------------------------------------------------
'''
array(["'eu'", "'e'", "'a'", "'u'", "'o'", "'ou'", "'au'", "'i'", "'en'",
       "'in'", "'ai'", "'un'", "'E'", "'ia'", "'one'", "'ei'", "'on'",
       "'io'", "'oi'", "'an'", "'O'", "'iu'", "'ien'", "'anu'", "'eni'",
       "'ui'", "'ein'", "'uin'", "'ua'", "'ian'", "'oua'", "'us'", "'oa'",
       "'Oi'", "'as'", "'ae'", "'Ou'", "'enin'", "'iO'", "'um'", "'aiu'",
       "'onu'", "'oni'", "'uni'", "'ain'", "'eo'", "'uo'", "'ea'",
       "'aun'", "'En'", "'oin'", "'ano'", "'anun'", "'ue'", "'uan'",
       "'ie'", "'iE'", "'Ea'", "'Oa'", "'ani'", "'Ei'", "'áu'", "'ao'",
       "'aue'"]
'''      
def FoneticaToTable(strVowel):
    sV = strVowel.replace('\'','')
    
    if sV == 'E':
        sV =  'ɛ'
    if sV == 'O':
        sV =  'ɔ'
    if sV == 'an':
        sV =  'ã'
    if sV == 'en':
        sV =  'ẽ'
    if sV == 'in':
        sV =  'ĩ'
    if sV == 'on':
        sV =  'õ'
    if sV == 'un':
        sV =  'ũ'
    if sV == 'ai':
        sV =  'aj'
    if sV == 'eu':
        sV =  'ew'
    if sV == 'oi':
        sV =  'oj'
    if sV == 'ou':
        sV =  'ow'
    if sV == 'au':
        sV =  'aw'
    if sV == 'ia':
        sV =  'ja'
    return '/{:}/'.format(sV)
# -----------------------------------------------------------------------------
# "a","á","à","â","ɐ","ã","e","é","ɛ","ê","i","ɪ","ɪ̃","í","ĩ","o","ó","ɔ","ô","õ","u","ú","ü","ʊ","ũ"
def vowelClass(strVowel):
    sV = strVowel.replace('\'','')
    
    lAAO = ("a","á","à","â")
    lAAN = ("ã")
    lSCO = ("ɐ")
    lSAO = ("e","é","ɛ","ê")
    lSPO = ("o","ó","ɔ","ô")
    lSPN = ("õ")
    lFAO = ("i","ɪ","í")
    lFAN = ("ɪ̃","ĩ")
    lFPO = ("u","ú","ü","ʊ")
    lFPN = ("ũ")
    if sV in lAAO:
        return ["aberta","anterior", "oral",1]
    if sV in lAAN:
        return ["aberta","anterior", "nasal",1]
    if sV in lSCO:
        return ["semi","central", "oral",1]
    if sV in lSAO:
        return ["semi","anterior", "oral",1]
    if sV in lSPO:
        return ["semi","posterior", "oral",1]
    if sV in lSPN:
        return ["semi","posterior", "nasal",1]
    if sV in lFAO:
        return ["fechada","anterior", "oral",1]
    if sV in lFAN:
        return ["fechada","anterior", "nasal",1] 
    if sV in lFPO:
        return ["fechada","posterior", "oral",1]
    if sV in lFPN:
        return ["fechada","posterior", "nasal",1]    
    return ["na","na","na",0]
# -----------------------------------------------------------------------------
'''
dt = ("eu", "ou", "au","ai", "ia","one", "ei","io","oi","iu", "ien","anu","eni","ui", 
"ein", "uin", "ua", "ian", "oua", "us", "oa","Oi", "as", "ae", "Ou", "enin", "iO",
 "aiu","onu", "oni", "uni", "ain", "eo", "uo", "ea","aun", "oin", "ano", "anun", 
 "ue", "uan","ie", "iE", "Ea", "Oa", "ani", "Ei", "áu", "ao","aue")
'''  
def diptongBinaryClass(strVowel):
    sV = strVowel.replace('\'','')
    lCrescente = ("ia","io","iu", "ua", "ian", "oua", "iO","uni",  "uo", "ea",
                  "ue", "uan","ie", "iE")
    lDecrescente = ("eu", "ou", "au","ai","one", "ei","oi", "ien","anu","eni",
                    "ui", "ein", "uin","us", "oa","Oi", "as", "ae", "Ou",
                    "enin","aiu","onu", "oni","ain", "eo","aun", "oin", "ano", 
                    "anun","Ea", "Oa", "ani", "Ei", "áu", "ao","aue")
    
    lAberto = ("eu", "ou", "au","ai","ei","oi","ui", "us","Oi", "as", "ae", "Ou", 
               "aiu", "eo","ea",  "Ea", "Oa", "Ei", "áu", "ao","aue")
    lFechado = ("one","ein", "uin", "ian", "ien","anu","eni", "enin","onu", 
              "oni", "uni", "ain","aun", "oin", "ano", "anun", "uan", "ani",
              "ia""io","iu""ua","oua", "oa", "iO", "uo", "ue", "ie", "iE")
    
    lOral = ("eu", "ou", "au","ai", "ia", "ei","io","oi","iu", "ui", "ua", 
             "oua", "us", "oa","Oi", "as", "ae", "Ou", "iO","aiu", "eo", "uo", 
             "ea", "ue", "ie", "iE", "Ea", "Oa", "Ei", "áu", "ao","aue")
    lNasal = ("one","ein", "uin", "ian", "ien","anu","eni", "enin","onu", 
              "oni", "uni", "ain","aun", "oin", "ano", "anun", "uan", "ani")
    
    
    tags = [ int((sV in lCrescente) and not (sV in lDecrescente)), 
             int((sV in lAberto) and not (sV in lFechado)), 
             int((sV in lOral) and not (sV in lNasal))]
    
    return tags
# "a","á","à","â","ɐ","ã","e","é","ɛ","ê","ẽ","i","ɪ","ɪ̃","í","ĩ","o","ó","ɔ","ô","õ","u","ú","ü","ʊ","ũ"
def vowelBinaryClass(strVowel):
    sV = strVowel.replace('\'','')
    # lVVV = ("a","á","à","â","ɐ","ã","e","é","ɛ","ê","i","ɪ","ɪ̃","í","ĩ","o","ó","ɔ","ô","õ","u","ú","ü","ʊ","ũ")
    lAberta = ("a","ã","á","à","â","ɐ","é","ɛ","ó","ɔ")
    lFechada = ("e","ê","ẽ","o","ô","i","ɪ","í","u","ú","ü","ʊ")
    lNasal = ("ã","ẽ","õ","ɪ̃","ĩ","ũ")
    lOral = ("a","á","à","â","ɐ","e","é","ɛ","ê","i","ɪ","í","o","ó","ɔ","ô","u","ú","ü","ʊ")
    lFrontal = ("ɪ̃","ĩ","ã","i","ɪ","í","e","ê","é","ɛ","ẽ","a","á","à","â","ɐ")
    lPosterior = ("o","ó","ɔ","ô","õ","u","ú","ü","ʊ","ũ")
    tags = [ int((sV in lAberta) and not (sV in lFechada)), 
             int((sV in lFrontal) and not (sV in lPosterior)), 
             int((sV in lOral) and not (sV in lNasal))]
    
    return tags
# -----------------------------------------------------------------------------
def consonantBinaryClass(strVowel):
    sV = strVowel.replace('\'','')
    # lPlosiva = ('b','c','d','g','k','p','t','q') #8
    
    # lPlosiva = ('b','c','d','g','k','p','t','q','r', 'ɾ') #10
    lFricativa = ('f','s','v','x','z','ʃ','ɣ', 'ʒ', 'ʤ', 'ʧ') #10
    lLiquidas = ('h','j','l','y','ʎ', 'w') #6
    # lLiquidas = ('h','j','l','y','ʎ', 'w','r', 'ɾ') #8
    
    lNasais = ('m','n','ɲ', 'ɳ', 'ŋ') #5
    # lOrais = 
    
    lLabio = ('b','f','m','p','v') # 5
    lAlveolo = ('d','j','l','n','r','s','t','z','ʃ', 'ɾ', 'ʒ', 'ʤ', 'ʧ') #13
    # lFundo = ('c','g','h','k','q','w','x','y','ʎ', 'ɲ', 'ɳ', 'ɣ', 'ŋ') #13
    
    lVozeada = ('b','d','g','h','j','l','m','n','r','v','w','y','z','ʎ', 'ɲ', 'ɳ', 'ɾ', 'ɣ', 'ʒ', 'ʤ', 'ŋ') #21
    # lDesvozeada = ('c','f','k','p','q','s','t','x','ʃ', 'ʧ') #10
    
    tags = [ int(sV in lVozeada), 
             int( (sV in lLabio) or (sV in lAlveolo)),
             int((sV in lFricativa) or (sV in lLiquidas)), 
             int(sV in lNasais)]
    
    return tags
# -----------------------------------------------------------------------------
'''
dt = ("eu", "ou", "au","ai", "ia","one", "ei","io","oi","iu", "ien","anu","eni","'ui'", 
"ein", "uin", "ua", "ian", "oua", "us", "oa","Oi", "as", "ae", "Ou", "enin", "iO",
 "aiu","onu", "oni", "uni", "ain", "eo", "uo", "ea","aun", "oin", "ano", "anun", 
 "ue", "uan","ie", "iE", "Ea", "Oa", "ani", "Ei", "áu", "ao","aue")
'''       
def soundBinaryClass(strSound):
    sV = strSound.replace('\'','')
    lstVowel = ("a","á","à","â","ɐ","ã","e","é","ɛ","ê","ẽ","i","ɪ","ɪ̃","í","ĩ","o","ó","ɔ","ô","õ","u","ú","ü","ʊ","ũ")
    lstConsonant = ('b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z','ʃ', 'ʎ', 'ɲ', 'ɳ', 'ɾ', 'ɣ', 'ʒ', 'ʤ', 'ʧ', 'ŋ')
    lstDiptongs = ("eu", "ou", "au","ai", "ia","one", "ei","io","oi","iu", "ien","anu","eni","'ui'", 
    "ein", "uin", "ua", "ian", "oua", "us", "oa","Oi", "as", "ae", "Ou", "enin", "iO",
     "aiu","onu", "oni", "uni", "ain", "eo", "uo", "ea","aun", "oin", "ano", "anun", 
     "ue", "uan","ie", "iE", "Ea", "Oa", "ani", "Ei", "áu", "ao","aue")
    is_vowel = (sV in lstVowel)
    is_Consonant = (sV in lstConsonant)
    is_Diptong = (sV in lstDiptongs)
    if (is_vowel and (not is_Consonant) and (not is_Diptong)):
        return [1, 1, *vowelBinaryClass(sV)]
    elif ((not is_vowel) and is_Consonant and (not is_Diptong)):
        return [0, *consonantBinaryClass(sV)]
    elif ((not is_vowel) and (not is_Consonant) and is_Diptong):
        return [0, 1, *diptongBinaryClass(sV)]
    else:
        return [0,0,0,0,0]
# -----------------------------------------------------------------------------    
# ('b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z','ʃ', 'ʎ', 'ɲ', 'ɳ', 'ɾ', 'ɣ', 'ʒ', 'ʤ', 'ʧ', 'ŋ')
def consonantClass(strVowel):
    sV = strVowel.replace('\'','')
    lPlosiva = ('b','c','d','g','k','p','t','q') #8
    
    lFricativa = ('f','s','v','x','z','ʃ','ɣ', 'ʒ', 'ʤ', 'ʧ') #10
    lLiquidas = ('h','j','l','r','y','ʎ', 'ɾ', 'w') #8
    
    lNasais = ('m','n','ɲ', 'ɳ', 'ŋ') #5
    
    lLabio = ('b','f','m','p','v') # 5
    lAlveolo = ('d','j','l','n','r','s','t','z','ʃ', 'ɾ', 'ʒ', 'ʤ', 'ʧ') #13
    
    lFundo = ('c','g','h','k','q','w','x','y','ʎ', 'ɲ', 'ɳ', 'ɣ', 'ŋ') #13
    
    lVozeada = ('b','d','g','h','j','l','m','n','r','v','w','y','z','ʎ', 'ɲ', 'ɳ', 'ɾ', 'ɣ', 'ʒ', 'ʤ', 'ŋ') #21
    lDesvozeada = ('c','f','k','p','q','s','t','x','ʃ', 'ʧ') #10
    
    tags = []
    if sV in lPlosiva:
        tags.append('plosiva')
    if sV in lFricativa:
        tags.append('fricativa')
    if sV in lLiquidas:
        tags.append('liquida')
    if sV in lNasais:
        tags.append('nasal')
    if len(tags) < 1:
        tags.append('na')
    if sV in lLabio:
        tags.append('labio')
    if sV in lAlveolo:
        tags.append('alveolo')
    if sV in lFundo:
        tags.append('fundo')
    if len(tags) < 2:
        tags.append('na')
    if sV in lVozeada:
        tags.append('vozeada')
    if sV in lDesvozeada:
        tags.append('desvozeada')
    if len(tags) < 3:
        tags.append('na')
    if 'na' in tags:
        tags.append(0)
    else:
        tags.append(1)
    return tags
# -----------------------------------------------------------------------------
def ajustVowelTag(tag):
    if (tag == 'an'):
        return 'ã'
    if (tag == 'en'):
        return 'ẽ'
    if (tag == 'in'):
        return 'ĩ'
    if (tag == 'un'):
        return 'ũ'
    if (tag == 'E'):
        return 'ɛ'
    if (tag == 'O'):
        return 'ɔ'
    if (tag == 'ai'):
        return 'aj'
    return tag
# -----------------------------------------------------------------------------
def ajustConsonantTag(vecTag):
    retTag = []
    nKaf = ["m","n","g","s","l","d", "p", "z", "v", "b", "k", "f", "t",
            "s", "x", "z","X", "o", "e", "NA"]
            #"e", "a", "u", "o", "i"]
           #  "'ɣ'", "'ʊ'", "'ɪ'",
           #  "'ʎ'", "'ʃ'",  "'ʧ'",
           # "'ʒ'", "'ɲ'", "'ɾ'", "'ʤ'"
           # "'en'","'in'", "'ai'",
           # "'un'", "'E'","'an'","'O'"
    for i in vecTag:
        iKey = i.replace("'","")
        if (iKey in  nKaf):
            retTag.append(iKey)
        else:
            if (iKey == 'ɣ'):
                retTag.append("")
            if (iKey == 'ʊ'):
                retTag.append("u")
            if (iKey == 'ɪ'):
                retTag.append("i")
            if (iKey == 'ʎ'):
                retTag.append("lh")
            if (iKey == 'ʒ'):
                retTag.append("z")
            if (iKey == 'ɲ'):
                retTag.append("nh")
            if (iKey == 'ɾ'):
                retTag.append("dz")
            if (iKey == 'ʤ'):
                retTag.append("nh")
            if (iKey == 'ʃ'):
                retTag.append("ss")
            if (iKey == 'ʧ'):
                retTag.append("th")
    return retTag
# -----------------------------------------------------------------------------
