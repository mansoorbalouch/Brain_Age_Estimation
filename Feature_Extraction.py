import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def Apply_Lin_Transform(X, technique, components, dir): 

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
        
def Apply_NonLin_Transform(X, technique):
    if (technique=="Kernel_PCA"):
        print("Kernel PCA performed")
    if (technique=="MDS"):
        # Matrix of Euclidean distances is converted into a centered Gram matrix, 
        # which can be directly used to perform PCA via eigendecomposition 

        print("Computing the Eucledian matrix!!")
        Comp_Dist_Mat(X)



"""
This function does the following task:
===============================================================
-> Takes a data matrix of size m * n where m rows represents the samples or points and 
    n columns represent the features or dimensions
-> Divides the data matrix into blocks or chunks and loads each block row wise
-> Computes the Eucledian and Cosine distance matrices of each block 
-> Returns the square distance matrices, each of the order m * m
"""
def Comp_Dist_Mat(X, no_of_blocks, rows, skip_rows, chunks, dir):
    m = len(X.rows)
    n = len(X.columns)
    # no_of_blocks = 13
    # rows = 250
    # skip_rows = 0
    # chunks = 10

    Euc_dist_mat = np.zeros((m,m))
    Cos_dist_mat = np.zeros((m,m))

    for i in range(6,no_of_blocks):
        rows = 250
        chunks = 10
        skip_rows = rows * i
        low_limit = rows * i
        if i==12:
            chunks=11
            rows = 209
            upper_limit = 209
        file = open(file_name)
        # load the ith diagonal data block
        data_mat_block_i = np.loadtxt(file ,delimiter = ",", usecols=np.arange(1,n), max_rows=rows, skiprows=skip_rows)
        # compute and store the distances and covariance blocks at the diagonals
        Euc_dist_mat[low_limit:low_limit+rows, skip_rows:rows+skip_rows]= euclidean_dist_compute(data_mat_block_i, data_mat_block_i, chunks)
        Cos_dist_mat[low_limit:low_limit+rows, skip_rows:rows+skip_rows]= cosine_dist_compute(data_mat_block_i, data_mat_block_i, chunks)
        file.close()

        for j in range(i+1,no_of_blocks): # loop till all upper triangular blocks are computed
            skip_rows = skip_rows + rows
            upper_limit = rows
            if j==12:
                rows = 209
                chunks=11
                upper_limit = 250
            else:
                rows=250
                chunks=10
            file = open(file_name)
            data_mat_block_j = np.loadtxt(file ,delimiter = ",", usecols=np.arange(1,3659573), max_rows=rows, skiprows=skip_rows)
            file.close()
            # compute and store the distances and covariance blocks in the upper triangle
            Euc_dist_mat[low_limit:low_limit+upper_limit, skip_rows:rows+skip_rows]= Euc_dist_mat(data_mat_block_i, data_mat_block_j, chunks)
            Cos_dist_mat[low_limit:low_limit+upper_limit, skip_rows:rows+skip_rows]= Cos_dist_mat(data_mat_block_i, data_mat_block_j, chunks)
            
        np.savetxt(dir+ "Euclidean_Dist_Mat_"+m+"x"+m+".csv", Euc_dist_mat, delimiter=",")
        np.savetxt(dir+ "Cosine_Dist_Mat_"+m+"x"+m+".csv", Cos_dist_mat, delimiter=",")

"""
-> Function definition for computing Eucledian distance matrix using dask delayed
Input: takes two blocks x and y 
"""
def euclidean_dist_compute(x, y, chunks):
    """Implementation using array concatenation"""
    values = [delayed(cdist)(x, xi, 'euclidean')
              for xi in np.split(y, chunks)]
    return np.concatenate(compute(*values, scheduler='threads'),
                          axis=1)

"""
-> Function definition for computing cosine distance matrix using dask delayed
Input: takes two blocks x and y 
"""
def cosine_dist_compute(x, y, chunks):
    """Implementation using array concatenation"""
    values = [delayed(cdist)(x, xi, 'cosine')
              for xi in np.split(y, chunks)]
    return np.concatenate(compute(*values, scheduler='threads'),
                          axis=1)
    



"""
############## Find EVD of the covariance matrix  ##################### 
"""
def Perform_EVD(X, chunks):
    cov_mat = cov_mat_compute(X.transpose(), X, chunks=chunks)
    D,Q = np.linalg.eig(cov_mat)

    fig = plt.figure(figsize=(10,5))
    eigen_values=D**2/np.sum(D**2)
    figure=plt.figure(figsize=(10,6))
    sing_vals=np.arange(3207) + 1
    plt.plot(sing_vals,D[2:3210], 'b', linewidth=2)
    plt.yscale("linear")
    plt.xscale("log")

    plt.title('Scree Plot')
    plt.xlabel('Principal Components')
    plt.ylabel('Eigenvalues')
    leg = plt.legend(['Eigenvalues from EVD'], loc='best', borderpad=0.3, 
                    shadow=False,
                    markerscale=0.4)
    leg.get_frame().set_alpha(0.4)

    plt.plot(45,2303245684.010627, 'o', color='black')
    plt.plot(73,366789333.1492814, 'o', color='black')

    # plt.show()
    # plt.savefig('/media/dataanalyticlab/Drive2/MANSOOR/Dataset/OpenBHB/EDA/Scree_plot_pca_openbhb.png')

"""
-> Function definition for computing covariance matrix using dask delayed
Input: takes a data matrix X of order m * n and its transpose X_t of order n * m
 -> m rows represent the samples or points and n columns represent the features or dimensions
 -> chunks represent the number of blocks that the data matrix needs to be divided into
Output: Returns the covariance matrix of order n * n
"""
def cov_mat_compute(X_t, X, chunks):
    """Implementation using array concatenation"""
    values = [delayed(np.cov)(X_t, X)]
    return np.concatenate(compute(*values, scheduler='threads'),
                          axis=1)