#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse

import warnings
warnings.simplefilter("ignore")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Perform PCA",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("comp", type=int, help="no of principle components")
parser.add_argument("src", help="Source location")
parser.add_argument("dest", help="Destination dir only")
parser.add_argument("dataset", type=str, help="Dataset name")

args = vars(parser.parse_args())

src = args["src"]
dest = args["dest"]
comp = args["comp"]
dataset = args["dataset"]

try:
    X = np.load(src, allow_pickle=True)
    ids = np.array([X[:,0]]).transpose()
    print("Input matrix has shape: ", X.shape)
    X = X[:,1:]
    X = StandardScaler().fit_transform(X)
    print("Matrix scaled! transforming to ", comp, "components..")
    if (comp < X.shape[1]):
        pca = PCA(n_components=int(comp))
    else:
        pca = PCA(n_components=X.shape[0]-1)
    X_pca = pca.fit_transform(X)
    print("Matrix transformed using PCA...")
    X_pca = np.append(ids,X_pca, axis=1)
    np.save(dest + "PCA_" + str(dataset) + "_" + str(comp) + "_comps.npy" ,X_pca)
    print("Reduced matrix saved and has size: ", X_pca.shape)
except:
    print("An exception occured!")