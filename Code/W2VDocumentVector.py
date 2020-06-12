#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 08:32:08 2020

@author: daniel
"""

import pandas as pd
import numpy as np
import warnings
from scipy.stats import kurtosis, skew
from gensim.models.word2vec import Word2Vec
import argparse
from tqdm import tqdm
import time

#from NLP_FeatureExtraction import 

class InvalidStatistic(Exception):
     def __init__(self, message = 'Invalid Statistic, check the list of allowed statistics'):
         self.message = message 
     def __str__(self):
         return repr(self.message)
#%%
def W2V_representation_by_sentence(sentence,Word2VecModel,UnknownWordsCero=True, dimensionModel=100):
    words_vector=sentence.split(" ")
    W2V_representation_sentece=[]
    word_not_found=0
    for word in words_vector:
        try:
            W2V_representation_sentece.append(Word2VecModel[word])              
        except Exception:
            word_not_found+=1
            if UnknownWordsCero:
                W2V_representation_sentece.append(np.zeros(dimensionModel))
            warnings.filterwarnings("ignore")
            pass 
    return W2V_representation_sentece,word_not_found


def statistics_from_document(word_embeddings, statistics = ['mean','std','kurtosis','skew','max','min']):
    
    statistical_dic={}
    if 'mean' in statistics:
        statistical_dic['mean'] = np.mean(word_embeddings, axis=0)
    if 'std' in statistics:
        statistical_dic['std'] = np.std(word_embeddings, axis=0)
    if 'kurtosis' in statistics:
        statistical_dic['kurtosis'] = kurtosis(word_embeddings, axis=0)
    if 'skew' in statistics:
        statistical_dic['skew'] = skew(word_embeddings, axis=0)
    if 'max' in statistics:
        statistical_dic['max'] = np.max(word_embeddings,axis=0)
    if 'min' in statistics:
        statistical_dic['min']=np.min(word_embeddings,axis=0)
        
    if sorted(list(statistical_dic.keys()))==sorted(statistics):
        vectorDocument=np.asarray(np.hstack(list(statistical_dic.values())))
        return vectorDocument
    else:
        raise InvalidStatistic
        
def DocumentVectorW2VStatistic(listDocuments, path_dir, Word2VecModelPath, statistics, dimensionModel, UnknownWordsCero=True):
    Word2VecModel=Word2Vec.load(Word2VecModelPath)
    documentsList=[]

    for i in tqdm(range(len(listDocuments))):
        word_embeddingsDocument, wordNotFound = W2V_representation_by_sentence(listDocuments[i], Word2VecModel, UnknownWordsCero, dimensionModel)
        statisticDoc=statistics_from_document(word_embeddingsDocument, statistics)
        documentsList.append(statisticDoc)
    np.savetxt(path_dir+'Features_W2V.txt',np.asarray(documentsList))


def Word2VecForDocuments(pathDocuments, pathDest, W2VmodalPath, statistics=['mean'], dimension = 100):
    listDocuments = list(pd.read_excel(pathDocuments)['Documents'])
    DocumentVectorW2VStatistic(listDocuments, pathDest, W2VmodalPath, statistics, dimension, False)    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("W2VModel", type=str, help="Path to the W2V model file.")
    parser.add_argument(
        "input_folder", type=str, help="Folder with the Documents NameColum->Documents"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Output folder. folder with the featues.",
    )
    args = parser.parse_args()

    Word2VecForDocuments(args.input_folder,args.output_folder,args.W2VModel)