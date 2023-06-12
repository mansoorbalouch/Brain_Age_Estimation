import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import *
import gc
import sklearn
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter


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


