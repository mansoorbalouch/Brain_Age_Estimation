import numpy as np
import pandas as pd
from math import *
import gc
import sklearn
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter
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


"""
Computes the feature selection metrics for given data matrix (feature matrix) and the target variable
--------------------------------------------
returns the mean, std, min, max, var, cov(x,y), corr(xy), entropy, MI and NMI
"""
def Comp_Feat_Sel_Metrics(X, Y):
    Index, mean_x, std_x, median, min_x, max_x, var_x, cov_xy, corr_xy, entropy_x, MI, NMI_score = []
    mu_y = np.mean(Y[:,])
    std_y = np.std(Y[:,])
    entropy_y = comp_entropy(Y[:,])
    for i in range(0, len(X.transpose())):
        Index.append(i)
        mu_x = np.mean(X[:,i])  # compute mean (x_i)
        mean_x.append(mu_x)
        st_x = np.std(X[:,i])  # compute std(x_i)
        std_x.append(st_x)
        min_x.append(np.min(X[:,i]))  # find min(x_i)
        max_x.append(np.max(X[:,i]))   # find max (x_i)
        var_x.append(comp_cov(X[:,i], X[:,i], mu_x, mu_x))  # compute var(x)
        cov = comp_cov(X[:,i], Y[:,], mu_x, mu_y)   # compute covariance(x_i, y)
        cov_xy.append(cov)
        if st_x != 0:
            corr_xy.append(cov/(st_x * std_y))       # compute correlation(x_i, y)
        else:
            corr_xy.append(0)
        # Entropy of feature x_i
        entropy_x.append(comp_entropy(X[:,i]))
        # Mutual Information computing the statistical dependence b/w X and Y
        MI.append(mutual_info_regression(X[:,i].reshape(-1,1),Y[:,]))
        # Normalized Mutual Information Score (b/w 0 and 1)
        # NMI = MI(X,Y)/mean{H(X), H(Y)}
        NMI_score.append(normalized_mutual_info_score(X[:,i], Y[:,]))
    feature_metrics = pd.DataFrame({"Id": Index, "Mean(x)": mean_x, "Std(x)": std_x, "Min(x)": min_x, "Max(x)": max_x, "Var(x)": var_x, 
                                    "Cov(xy)": cov_xy, "Corr(xy)": corr_xy, "Entropy(x)": entropy_x, "MI(xy)": MI, "NMI(xy)": NMI_score}).transpose()
    return feature_metrics



# compute covariance between x and y
def comp_cov(x, y, m1, m2):
    return sum([(xi - m1) * (yi - m2) for xi, yi in zip(x, y)]) / (len(x))

def comp_entropy(values):
    values = values.astype(int)
    counts = np.bincount(values)
    probs = counts[np.nonzero(counts)] / float(len(values))
    return - np.sum(probs * np.log(probs))


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
        Euc_dist_mat[low_limit:low_limit+rows, skip_rows:rows+skip_rows]= Euc_dist_mat(data_mat_block_i, data_mat_block_i, chunks)
        Cos_dist_mat[low_limit:low_limit+rows, skip_rows:rows+skip_rows]= Cos_dist_mat(data_mat_block_i, data_mat_block_i, chunks)
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
    
