#!/usr/bin/env python

import warnings
warnings.simplefilter("ignore")

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np
import math
import sys
from tensorflow.keras.models import load_model
import random
import glob
import re
from collections import defaultdict
from tensorflow.keras.optimizers import Adam, SGD,Adagrad
from tensorflow.compat.v1 import reset_default_graph
import ResNet
from ResNet import generateAgePredictionResNet
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping 
from tensorflow.python.keras import backend as K
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import load_model
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
from argparse import ArgumentParser


# Parse command line arguments
parser = argparse.ArgumentParser(description="Build and train a ResNet model for brain age prediction",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("src", help="Source location")
parser.add_argument("dataShape", type=tuple, help="Shape of a MRI [default 121*145*121]", nargs='?', default=(121, 145, 121), const=(121, 145, 121))
parser.add_argument("epochs", type=int, help="No of epochs")

args = vars(parser.parse_args())
src = args["src"]
dataShape = args["dataShape"]
nEpochs = args["epochs"]

# dataShape = (121, 145, 121)

############## load the data matrix of size (m,i,j,k,c) where m is the number of samples,
#   i,j,k represent the 3D dimensions of the MRI, and c represents the channels ############    
X = np.load(src + "VBM_OpenBHB/VBM_OpenBHB.npy")
Y = pd.read_csv(src + "SubInfoOpenBHB.csv")
print("Data matrix loaded... shape: ", X.shape )


################## train-test slpit ###############
################## split:1
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X[:int(len(X)/2),:,:,:,:], Y.loc[:int(len(Y)/2)-1,"age"], test_size=0.2, random_state=42)
print("1st train-test split completed... train shape: ", X_train_1.shape, "test shape: ", X_test_1.shape)
np.save(src + "VBM_OpenBHB/VBM_OpenBHB_Train1.npy" ,X_train_1)
np.save(src + "VBM_OpenBHB/VBM_OpenBHB_Test1.npy" ,X_test_1)
del X_train_1, X_test_1
print("Split subset 1 saved...")

################ split: 2 
#### free the first half of the data matrix
X = X[int(len(X)/2):,:,:,:,:] 
Y = Y.loc[int(len(Y)/2):,"age"]
X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(X, Y, test_size=0.2, random_state=42)
print("2st train-test split completed... train shape: ", X_train_2.shape, "test shape: ", X_test_2.shape)
np.save(src + "VBM_OpenBHB/VBM_OpenBHB_Train2.npy" ,X_train_2)
np.save(src + "VBM_OpenBHB/VBM_OpenBHB_Test2.npy" ,X_test_2)

############### merge the two sets
del X

print("Loading split subset 1...")
X_train_1 = np.load(src + "VBM_OpenBHB/VBM_OpenBHB_Train1.npy")
X_test_1 = np.load(src + "VBM_OpenBHB/VBM_OpenBHB_Test1.npy")

X_train = np.concatenate((X_train_1, X_train_2))
Y_train = np.concatenate((Y_train_1, Y_train_2))
del X_train_1, X_train_2
X_test  = np.concatenate((X_test_1, X_test_2))
Y_test = np.concatenate((Y_test_1, Y_test_2))
del X_test_1,X_test_2 
print("Train-test merge completed... new train shape: ", X_train.shape, "test shape: ", X_test.shape)

############ model hyperparameters #################
batchSize = 4
steps_per_epoch= X_train.shape[0]//batchSize
validation_steps = X_test.shape[0]//batchSize
default_parameters = [0.001,1e-6,'RawImg','IncludeGender','IncludeScanner',0.00005,0.2,40,10]
lr, decayRate, meanImg, gender, scanner,regAmount, dropRate, maxAngle,maxShift = default_parameters

######### build model #############
model = generateAgePredictionResNet(dataShape,regAmount=regAmount,dropRate=dropRate)
lr=0.001
#decayRate=1e-6
#momentum=0.9
adam = Adam(learning_rate=lr)
model.compile(loss='mean_absolute_error',optimizer=adam, metrics=['mae','mse'])
print("Model compiled...")

######### save the best model ########
mc = ModelCheckpoint(src+'/Models/BrainAgeResNet-OpenBHB_VBM.h5',verbose=1,mode='min',save_best_only=True)
early = EarlyStopping(patience=100, verbose=1)

print("Fitting the Model...")
######### train the model ##############
h = model.fit(x=X_train, y=Y_train,
                        validation_steps=validation_steps,
                        steps_per_epoch=steps_per_epoch, 
                        epochs=nEpochs,
                        verbose=1,
                        max_queue_size=32,
                        workers=4,
                        use_multiprocessing=False,
                        callbacks=[mc,early]
                           )
print("Model fitted...")

########## save the model details (weights, architecture, compilaion details) ############
model.save(src + '/Models/BrainAgeResNet-OpenBHB_VBM-TrainedFor{}Epochs.h5'.format(len(h.history['loss'])))
print("Model saved...")

######### load the best model and perform predictions on the val and test set #######
model = load_model(src + 'Models/BrainAgeResNet-OpenBHB_VBM-TrainedFor{}Epochs.h5'.format(len(h.history['loss'])))
# val_prediction = model.predict(X_val,
#                         verbose=1,
#                         max_queue_size=32,
#                         workers=4,
#                         use_multiprocessing=False
#                         )

test_prediction = model.predict(X_test,
                        verbose=1,
                        max_queue_size=32,
                        workers=4,
                        use_multiprocessing=False
                        )

model_performance = model.evaluate(X_test, Y_test)
print(model_performance)
# train_predictions[:,0].save(src + "Y_val_pred-BrainAgeResNet")
np.save(src + "Y_test_pred-BrainAgeResNet.npy", test_prediction[:,0])


