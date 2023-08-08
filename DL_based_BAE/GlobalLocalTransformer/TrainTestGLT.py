##### dependencies ###
# pytorch
# numpy
# sklearn
#####################

from GlobalLocalTransformer import *
import torch 
import torch.nn as nn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")

class Trainer:
    """
    The simple trainer.
    """
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
            
    # s_start = starting slice, s_end = ending slice
    def dataLoaderGLT(self, source_path, s_start, s_end,  labels_path=""):
        if (len(labels_path)>1):
            X = np.load(labels_path + "VBM_OpenBHB.npy")
        else:
            X = np.load(source_path + "VBM_OpenBHB.npy")
        Y = pd.read_csv(source_path + "SubInfoOpenBHB.csv")
        print("Data tensor loaded... shape: ", X.shape )
        labels = Y.loc[:,"age"]
        X_s = X[:,:,:,s_start:s_end,0]

        X_train, X_test, Y_train, Y_test = train_test_split(X_s, labels, test_size=0.2, random_state=42)

        X_train = torch.from_numpy(X_train)
        X_train = X_train.permute(0,3,1,2)

        X_test = torch.from_numpy(X_test)
        X_test = X_test.permute(0,3,1,2)
        print("Training set size: ", X_train.shape, ", Test set size: ", X_test.shape)

        Y_train = torch.tensor(np.array(Y_train))
        Y_test = torch.tensor(np.array(Y_test))
        # torch.tensor()

        # print(Y_train.shape, Y_test.shape)
        return X_train, X_test,  Y_train, Y_test
  

    def trainGLT(self, attention_model, regression_model, X_train, X_test, Y_train, Y_test,  criterion, optimizer, n_epochs, batch_size):
        self.attention_model = attention_model
        self.regression_model = regression_model
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_loss = [] 
        self.val_acc = []
        self.train_dataloader = DataLoader(list(zip(X_train, Y_train)), batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(list(zip(X_test, Y_test)), batch_size=batch_size, shuffle=True)
        

        # Loop through the number of epochs
        for self.epoch in range(n_epochs):
            self.train_epoch_loss = 0  # Initialize loss for the entire epoch
            self.test_epoch_mae = 0   # Initialize the total MAE for the entire epoch
            optimizer.zero_grad()  # a clean up step for PyTorch

            loss = trainEpoch(self)
            self.train_loss .append(loss)

            
            mae = evaluateGLT(self)
            self.val_acc.append(mae)
            
        return self.train_loss , self.val_acc
    

def trainEpoch(self):
    """
    Training loop -> train the model using the entire training set for one complete epoch 
    """   

    # Loop through the batches and update the gradients after each batch
    for X_batch, y_batch in self.train_dataloader:
        self.regression_model.train()
        zlist = self.attention_model(X_batch)             # forward pass
        output_tensors = torch.cat(zlist, dim=1)
        self.batch_loss = 0
        for i in range(len(X_batch)):
            output = self.regression_model(output_tensors[i])
            loss = self.criterion(output, y_batch[i])  # Compute the loss for each sample
            self.batch_loss += loss  # Accumulate the loss for the entire batch
            self.train_epoch_loss += loss  # Accumulate the loss for the entire epoch
            
        batch_avg_loss = self.batch_loss / len(X_batch)  # Calculate the average loss for the batch
        batch_avg_loss.backward(retain_graph=True)  # Backward pass (compute parameter updates)
        # print("Batch completed")
        self.optimizer.step()


    train_epoch_avg_loss = round((self.train_epoch_loss / len(self.train_dataloader.dataset)).item(),3)
    print("Epoch:", self.epoch, "Loss:", train_epoch_avg_loss)
    return train_epoch_avg_loss


def evaluateGLT(self):
    """
    Validation loop -> evaluate the model on the entire test set for one epoch
    """

    self.regression_model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # loop through each batch of the test set and compute the evaluation metrics
        for X_batch, y_batch in self.test_dataloader:
            total_samples = 0
            zlist = self.attention_model(X_batch)             # forward pass
            output_tensors = torch.cat(zlist, dim=1)
            self.batch_mae = 0
            for i in range(len(X_batch)):
                output = self.regression_model(output_tensors[i])
                mae = torch.mean(torch.abs(output - y_batch[i]))
                self.test_epoch_mae += mae.item() # Accumulate the loss for the entire epoch
                self.batch_mae += mae.item() # Accumulate the loss for the entire batch
                
            avg_mae = self.batch_mae / len(X_batch)  # Calculate the average MAE for the epoch
            
        test_epoch_avg_mae = round((self.test_epoch_mae/len(self.test_dataloader.dataset)),3)
        print("Epoch:", self.epoch, "MAE: ", test_epoch_avg_mae)
        return test_epoch_avg_mae


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create the model, optimizer, loss function and trainer
    attention_model = GlobalLocalBrainAge(5, patch_size=32, step=32,
                        nblock=6, backbone='vgg8')
    regression_model = nn.Linear(13, 1)
    attention_model = attention_model.double()
    regression_model = regression_model.double()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(regression_model.parameters(), lr=0.0002)
    src = "/media/dataanalyticlab/Drive2/MANSOOR/Dataset/Test/"
    trainer = Trainer(attention_model, optimizer, criterion, device=device)
    X_train, X_test,  Y_train, Y_test = trainer.dataLoaderGLT(src, 55,60,src)
    trainer.trainGLT(attention_model, regression_model,X_train, X_test,  
                     Y_train, Y_test,criterion, optimizer, n_epochs=20, batch_size=2)


if __name__ == '__main__':
    main()