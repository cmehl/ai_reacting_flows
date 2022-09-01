#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 18:37:28 2021

@author: mehlc
"""

import pandas as pd 
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_pca(dtb_path, k):
    
    # Dataframe
    df = pd.read_csv(filepath_or_buffer=dtb_path, sep=';')
    
    # Extracting required features
    features = df.columns.to_list()
    features.remove('Time')
    features.remove('Simulation number')
    features.remove('Pressure')
    x = df.loc[:, features].values
    
    # Scaling data
    pca_scaler = StandardScaler()
    pca_scaler.fit(x)
    x = pca_scaler.transform(x)
    
    # Performing PCA
    pca_algo = PCA(n_components=k)
    pca_algo.fit(x)
    principalComponents = pca_algo.transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = [f'Principal Component {k}' for k in range(1,k+1)] )
    
    # Adding variable to color (here: temperature)
    df_train_data_pca = pd.concat([principalDf, df[['Temperature']]], axis = 1)
    
    
    return pca_algo, df_train_data_pca, pca_scaler
    

