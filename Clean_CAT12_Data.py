import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import Regression_tasks
import EDA_Visualization
import re
import glob
import os


vol_files = {
        1: "ROI_cobra_Vgm",
        2: "ROI_cobra_Vwm",
        3: "ROI_neuromorphometrics_Vcsf",
        4: "ROI_neuromorphometrics_Vgm",
        5: "ROI_neuromorphometrics_Vwm",
        6: "ROI_suit_Vgm",
        7: "ROI_suit_Vwm",
        8: "ROI_thalamic_nuclei_Vgm" } 

thick_files = ["ROI_aparc_DK40_thickness", "ROI_aparc_a2009s_thickness"]

"""
This function does the following task:
-> load the CAT12 data files and clean the data matrix by removing the redundant/NaN features (columns) 
--------------------------------------------------------------------------------
Input: takes the path of the directory with the CAT12 generated measurements
Output: returns three clean data matrices having all the volume, thickness, and combined data 
"""
def Rem_ROI_Features(dataset, dir):
   # load and clean the ROI thickness features
    ROI_aparc_DK40_thickness = pd.read_csv(dir + dataset +"_SBM_thickness/" + thick_files[0] + ".csv")
    ROI_aparc_a2009s_thickness = pd.read_csv(dir + dataset +"_SBM_thickness/" + thick_files[1] + ".csv")

    ROI_aparc_DK40_thickness = ROI_aparc_DK40_thickness.drop(["lunknown", "runknown","lcorpuscallosum", "rcorpuscallosum"], axis=1)
    ROI_aparc_a2009s_thickness = ROI_aparc_a2009s_thickness.drop(["lMedial_wall","rMedial_wall" ], axis=1)

    # load and clean the ROI voxel-based volume features
    ROI_cobra_Vgm = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[1] + ".csv")
    ROI_cobra_Vwm = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[2] + ".csv")
    ROI_neuromorphometrics_Vcsf = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[3] + ".csv")
    ROI_neuromorphometrics_Vgm = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[4] + ".csv")
    ROI_neuromorphometrics_Vwm = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[5] + ".csv")
    ROI_suit_Vgm = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[6] + ".csv")
    ROI_suit_Vwm = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[7] + ".csv")
    ROI_thalamic_nuclei_Vgm = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[8] + ".csv")


    ROI_neuromorphometrics_Vcsf = ROI_neuromorphometrics_Vcsf.drop(["Right Pallidum","Left Pallidum", "Left vessel", "Right vessel"], axis=1)
    ROI_neuromorphometrics_Vgm  = ROI_neuromorphometrics_Vgm.drop(["Left vessel", "Right vessel"], axis=1)
    ROI_neuromorphometrics_Vwm  = ROI_neuromorphometrics_Vwm.drop(["Left vessel", "Right vessel"], axis=1)
    ROI_thalamic_nuclei_Vgm = ROI_thalamic_nuclei_Vgm.drop(["lAnteroventral", "lVentral_Lateral_Anterior"], axis=1)

    ######### concatenate all the ROI volumetric features and save to local dir ########

    CAT12_ROI_Vol = pd.concat([ROI_cobra_Vgm, ROI_cobra_Vwm.iloc[:,1:], ROI_neuromorphometrics_Vcsf.iloc[:,1:], 
                        ROI_neuromorphometrics_Vgm.iloc[:,1:], ROI_neuromorphometrics_Vwm.iloc[:,1:], ROI_suit_Vgm.iloc[:,1:],
                        ROI_suit_Vwm.iloc[:,1:], ROI_thalamic_nuclei_Vgm.iloc[:,1:]], axis=1)

    ######### concatenate all the ROI thickness features ######### 
    CAT12_ROI_Thick = pd.concat([ROI_aparc_DK40_thickness, ROI_aparc_a2009s_thickness.iloc[:,1:]], axis=1)

    ######## concatenate the volumetric and thickness features ###########
    CAT12_ROI_Vol_Thick = pd.concat([CAT12_ROI_Vol.iloc[:,:-1], CAT12_ROI_Thick.iloc[:,1:].set_index(CAT12_ROI_Thick.index)], axis=1)

    return CAT12_ROI_Thick, CAT12_ROI_Vol, CAT12_ROI_Vol_Thick


"""
This function does the following task:
-> Perform image quality control of the preprocessed dataset 
----------------------------------------------------------------------
Input: takes a data matrix of order m*n and path the CAT12 quality reports and the IQR criteria
Output: returns the data matrix with k rows after applying the IQR criteria where k<=m
"""
def Perform_CAT12_QC(dataMatrix, data_dir, dataset, IQR_criteria):
    report_files = glob.glob(os.path.join(data_dir + "report/", "catlog_*.txt"))
    IQR =[]
    ids = []
    substr = "Image Quality Rating (IQR):"
    QC_dir = dataset+"CAT12_QC/"
    for report in report_files:
        mylines = []                            
        linenum = 0
        # make a list of the IQR values
        with open (report, 'rt') as myfile:    
            for line in myfile:                  
                linenum += 1
                if line.find(substr) != -1:    # if case-insensitive match
                    iqr = re.findall(r'\d+', line)
                    iqr = float(str(iqr[0]+"."+iqr[1]))
                    IQR.append(iqr)
    X_IQR = pd.concat([dataMatrix.iloc[:,0], pd.Series(IQR)], axis=1)
    # apply the quality control criteria on the IQR
    X_IQR_QC = X_IQR.drop(X_IQR[X_IQR.iloc[:,1]<IQR_criteria].index)
    merged = dataMatrix.merge(X_IQR_QC, left_on="names", right_on="names", how="inner")
    X_CAT12_QC = merged.drop(columns=0)

    if (os.path.exists(os.path.join(data_dir, QC_dir))==False):
        os.mkdir(os.path.join(data_dir, QC_dir))
    
    # convert to numpy array and save
    size = np.shape(X_CAT12_QC)
    np_CAT12_QC = X_CAT12_QC.to_numpy()
    np.save(data_dir+ QC_dir+"Prep_CAT12_Vol_"+dataset+"_QC_"+str(size[0])+"x"+str(size[1]-2)+".npy", np_CAT12_QC  )
    X_CAT12_QC.to_csv(data_dir+ QC_dir+"Prep_CAT12_Vol_"+dataset+"_QC_"+str(size[0])+"x"+str(size[1]-2)+".csv")
    return X_CAT12_QC