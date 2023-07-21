import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import os
import glob
import re


dataShape = (121, 145, 121)
vbm_dir = "/home/imdad.khan/brain_age_mansoor/OpenBHB/VBM_OpenBHB"
vbm_files = glob.glob(os.path.join(vbm_dir, "*.npy"))
metadata_path = "/home/imdad.khan/brain_age_mansoor/OpenBHB/participants_openbhb.csv"

# X = np.empty((3966,dataShape[0],dataShape[1],dataShape[2],1))
# Ids = []
# for i in range(len(vbm_files)):
#     id_i = re.search("sub-(.*)_prep", str(vbm_files[i])).group(1)
#     X[i,:,:,:,:] = np.load(vbm_files[i]).reshape(dataShape+(1,))
#     Ids.append(id_i)
#     if (i%100==0):
#         print(i," files loaded...")
# print("Data matrix created with shape: ", X.shape)
# np.save(vbm_dir+"/VBM_OpenBHB.npy",X)
# np.save(vbm_dir+"/IDs_OpenBHB.npy",Ids)

# print("Data matrix saved successfully...")

####### takes the IDs and the path to the metadata file ########
def getOpenBHBLabels(IDs, metadata_path):
    metadata = pd.read_csv(metadata_path, sep="\t")
    if (type(IDs)!="pandas.core.frame.DataFrame"):
        IDs = pd.DataFrame({"participant_id":Ids}, dtype=float)
    merged = IDs.merge(metadata, left_on="participant_id", right_on="participant_id", how="inner")
    # return a dataframe with the IDs, Age, and Sex variables of the subjects
    return merged.loc[:,["participant_id", "age", "sex"]]


Ids = np.load(vbm_dir+"/IDs_OpenBHB.npy")
OpenBHB_Metadata = getOpenBHBLabels(Ids,metadata_path)
OpenBHB_Metadata.to_csv("/home/imdad.khan/brain_age_mansoor/OpenBHB/SubInfoOpenBHB.csv")