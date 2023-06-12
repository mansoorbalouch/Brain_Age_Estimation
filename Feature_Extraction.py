import numpy as np
import pandas as pd
import os
import io
import glob
import sklearn
import re
from sklearn.cluster import KMeans
from csv import writer
from csv import reader 
from scipy.spatial import distance_matrix
from numpy import genfromtxt
import dask.dataframe as dd
import dask.array as da
from scipy.spatial import distance
import gc
from scipy.spatial.distance import cdist
from dask import compute, delayed
import dask
import  dask_ml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn import random_projection
from sklearn.random_projection import johnson_lindenstrauss_min_dim


def Reduce_Dim(X, technique, components, dir): 

    ######## RBF sampler (random Fourier features)   #####################
    if (technique=="RFF"):
        rbf = RBFSampler(gamma=1, n_components=components)
        rbf.fit(X)
        X_reduced = rbf.transform(X)
        X_reduced = pd.DataFrame(X_reduced)
        print("Matrix transformed using RFF")
        X_reduced.to_csv(dir+ "_reduced_kernel_trick_" + components + "_comp.csv")
        X_reduced.to_numpy(dir+ "_reduced_kernel_trick_" + components + "_comp.npy")
        print("Reduced matrix file saved")
        return X_reduced

    #########  PCA-based transformations #######################
    if (technique=="PCA"):
        # data scaling
        X_scaled = StandardScaler().fit_transform(X)
        print("Matrix scaled")
        # del X
        # gc.collect()
        pca = PCA(n_components=components)
        X_pca_features = pca.fit_transform(X_scaled)
        print("Matrix transformed using PCA")
        X_pca_features.to_csv(dir+ "_PCA_" + components + "_comp.csv")
        np.save(dir+ "_PCA_" + components + "_comp.npy", X_pca_features)
        print("Reduced matrix file saved")
        return X_pca_features


    ########### JL transform #################
    if (technique=="JL"):
        n = len(X)
        # calculate mininimum number of random projection required by JL
        min_dim = johnson_lindenstrauss_min_dim(n_samples=n, eps=0.1)
        # perform random projection
        transformer = random_projection.GaussianRandomProjection(n_components=min_dim)
        X_JL_transformed = transformer.fit_transform(X)
        print("Data transformed using JL transform, now saving reduced data!!!")
        np.save(dir+ "_JL_transformed_" + components + "_comp.npy", X_JL_transformed)
        X_JL_transformed.to_csv(dir+ "_JL_transformed_" + components + "_comp.csv")
        print("Reduced matrix file saved")
        return X_JL_transformed
        
