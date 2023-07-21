import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nb
import seaborn as sns
from nibabel.gifti import gifti
import nipype as nip
from nipype.interfaces import cat12, matlab, spm, freesurfer
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
    ROI_aparc_DK40_thickness = pd.read_csv(dir + dataset +"_SBM_thickness/" + thick_files[0] + ".csv")   ### n = 68
    ROI_aparc_a2009s_thickness = pd.read_csv(dir + dataset +"_SBM_thickness/" + thick_files[1] + ".csv") ### n = 150

    ROI_aparc_DK40_thickness = ROI_aparc_DK40_thickness.drop(["lunknown", "runknown","lcorpuscallosum", "rcorpuscallosum"], axis=1)
    ROI_aparc_a2009s_thickness = ROI_aparc_a2009s_thickness.drop(["lMedial_wall","rMedial_wall" ], axis=1)

    # load and clean the ROI voxel-based volume features
    ROI_cobra_Vgm = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[1] + ".csv")                  ### n = 52 
    ROI_cobra_Vwm = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[2] + ".csv")                  ### n = 52
    ROI_neuromorphometrics_Vcsf = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[3] + ".csv")    ### n = 132
    ROI_neuromorphometrics_Vgm = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[4] + ".csv")     ### n = 134
    ROI_neuromorphometrics_Vwm = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[5] + ".csv")     ### n = 134
    ROI_suit_Vgm = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[6] + ".csv")                   ### n = 28
    ROI_suit_Vwm = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[7] + ".csv")                   ### n = 28
    ROI_thalamic_nuclei_Vgm = pd.read_csv(dir + dataset +"_VBM_volume/" + vol_files[8] + ".csv")        ### n = 20


    ROI_neuromorphometrics_Vcsf = ROI_neuromorphometrics_Vcsf.drop(["Right Pallidum","Left Pallidum", "Left vessel", "Right vessel"], axis=1)
    ROI_neuromorphometrics_Vgm  = ROI_neuromorphometrics_Vgm.drop(["Left vessel", "Right vessel"], axis=1)
    ROI_neuromorphometrics_Vwm  = ROI_neuromorphometrics_Vwm.drop(["Left vessel", "Right vessel"], axis=1)
    ROI_thalamic_nuclei_Vgm = ROI_thalamic_nuclei_Vgm.drop(["lAnteroventral", "lVentral_Lateral_Anterior"], axis=1)  ### it has an additional column "rHabenula"

    ######### concatenate all the ROI volumetric features ########

    CAT12_ROI_Vol = pd.concat([ROI_cobra_Vgm, ROI_cobra_Vwm.iloc[:,1:], ROI_neuromorphometrics_Vcsf.iloc[:,1:], 
                        ROI_neuromorphometrics_Vgm.iloc[:,1:], ROI_neuromorphometrics_Vwm.iloc[:,1:], ROI_suit_Vgm.iloc[:,1:],
                        ROI_suit_Vwm.iloc[:,1:], ROI_thalamic_nuclei_Vgm.iloc[:,1:]], axis=1)

    ######### concatenate all the ROI thickness features ######### 
    CAT12_ROI_Thick = pd.concat([ROI_aparc_DK40_thickness, ROI_aparc_a2009s_thickness.iloc[:,1:]], axis=1)

    ######## concatenate the volumetric and thickness features ###########
    CAT12_ROI_Vol_Thick = pd.concat([CAT12_ROI_Vol.iloc[:,:-1], CAT12_ROI_Thick.iloc[:,1:].set_index(CAT12_ROI_Thick.index)], axis=1)

    return CAT12_ROI_Thick, CAT12_ROI_Vol, CAT12_ROI_Vol_Thick


"""
Input: this function takes the path of the smoothed MRI nifti files and the tissue type
Output: returns a table m*n containing the voxel intensity values where each row represents a subject
    and columns represent the voxel intentities
"""
def Load_Voxel_Features(MRI_Dir, source, tissue_type, sav_loc=False):
    # get all the specified tissue nifti files
    if (tissue_type=="GM"):
        t_id = 1
    elif (tissue_type=="WM"):
        t_id = 2
    elif (tissue_type=="CSF"):
        t_id = 3
    smoothed_files = glob.glob(os.path.join(MRI_Dir + "mri/", "s6mwp"+str(t_id)+"*.nii"))
    voxel_intensity_mat = []
    for i in range(len(smoothed_files)):
        smwp = nb.load(smoothed_files[i])
        # extract the SubjectID from the filename
        if (source=="IXI"):
            id_i = re.search(r's6mwp'+str(t_id)+'IXI(.*?)-', smoothed_files[i])
        elif (source=="ADNI"):
            id_i = re.search(r's6mwp'+str(t_id)+'ADNI_(.*?).nii', smoothed_files[i])
        elif (source=="OASIS"):
            id_i = re.search(r's6mwp'+str(t_id)+'(.*?).nii', smoothed_files[i])
        elif (source=="ICBM"):
            id_i = re.search(r's6mwp'+str(t_id)+'ICBM_(.*?)_MR', smoothed_files[i])
        # check if the string search didn't return anything (NoneType)
        if id_i is None:
            print("ID not matched with the given pattern!!")
            continue
        else:
            id_i =  id_i.groups(1)[0]
        # convert Nifti file to a numpy array and flatten to a 1D arr
        a = np.array(smwp.dataobj).flatten()
        # insert ID at the front (0th column index) of the ith swmp row
        a = np.insert(a.astype(object), 0, str(id_i), axis=0)
        voxel_intensity_mat.append(a)
    print("Data matrix created. Converting to df...")
    voxel_intensity_mat = np.array(voxel_intensity_mat)
    print(source + " voxel data matrix has shape: " + str(voxel_intensity_mat.shape))
    # rename the column name (which already has no name)
    voxel_intensity_mat = voxel_intensity_mat.rename(columns={voxel_intensity_mat.columns[0]: "Subject_ID"})
    if (sav_loc==True):
        vox_dir = "Voxel_Features/"
        # check if the specified dir exists, otherwise create it
        if (os.path.exists(os.path.join(MRI_Dir, vox_dir))==False):
            os.mkdir(os.path.join(MRI_Dir, "Voxel_Features/"))
        # convert to numpy array and save
        size = np.shape(voxel_intensity_mat)
        np.save(MRI_Dir+ vox_dir+source+"_Vox_CAT12_" + str(size[0])+"x"+str(size[1]-2)+".npy", voxel_intensity_mat )
        print("Np array saved to the local dir!!")
    return voxel_intensity_mat



def Load_Surf_Features(Data_dir):
    # # Load the thickness data
    # # file_path = "path_to_file/h.thickness"
    # fs2nii = freesurfer.MRIsConvert()
    # fs2nii.inputs.in_file = thick_in_file
    # # fs2nii.inputs.out_datatype = "nii"
    # fs2nii.run()

    # read files (with no file extensions) such as thickness, sulci depth etc. 
    # and convert to an array with continuous values

    # thick = np.fromfile(thick_in_file, dtype=np.float32)
    # depth = np.fromfile(depth_in_file, dtype=np.float32)
    # gyrification = np.fromfile(gyrf_in_file, dtype=np.float32)


    # print(thick.shape, depth.shape, gyrification.shape)
    # data = nb.freesurfer.load(thick_in_file)

    # # Access the cortical thickness data array
    # thickness_data = data.get_fdata()

    # # If necessary, transpose or manipulate the data
    # thickness_data = np.transpose(thickness_data)
    # Template surface files
    dkt_atlas_dir = "/media/dataanalyticlab/Drive2/MATLAB/spm12/toolbox/cat12/atlases_surfaces"
    lh_atlas = os.path.join(dkt_atlas_dir,'lh.aparc_a2009s.freesurfer.annot')
    rh_atlas = os.path.join(dkt_atlas_dir, 'rh.aparc_a2009s.freesurfer.annot')

    surf_files = [os.path.join(Data_dir,"surf",'lh.sphere.reg.ADNI_002_S_0295_T1w.gii'), os.path.join(ADNI_data_dir,"surf",'lh.sphere.reg.ADNI_002_S_0295_T1w.gii'), 
                os.path.join(Data_dir,"surf",'lh.sphere.ADNI_002_S_0295_T1w.gii'), os.path.join(ADNI_data_dir,"surf",'rh.sphere.ADNI_002_S_0295_T1w.gii'), 
                os.path.join(Data_dir,"surf",'lh.central.ADNI_002_S_0295_T1w.gii'), os.path.join(ADNI_data_dir,"surf",'rh.central.ADNI_002_S_0295_T1w.gii'), 
                os.path.join(Data_dir,"surf",'lh.pbt.ADNI_002_S_0295_T1w'), os.path.join(ADNI_data_dir,"surf",'rh.pbt.ADNI_002_S_0295_T1w')]
    lh_measure = os.path.join(Data_dir,"surf",'lh.thickness.ADNI_002_S_0295_T1w')
    extract_additional_measures = cat12.surface.ExtractROIBasedSurfaceMeasures(surface_files=surf_files, lh_surface_measure=lh_measure, lh_roi_atlas=lh_atlas, rh_roi_atlas=rh_atlas)
    extract_additional_measures.run(cwd="/media/dataanalyticlab/Drive2/MANSOOR/Dataset/Preprocessed_MRIs/CAT12_VBM/Test_surface/") 


"""
This function does the following task:
-> Perform image quality control of the preprocessed dataset 
----------------------------------------------------------------------
Input: takes a data matrix of order m*n and path the CAT12 quality reports and the IQR criteria
Output: returns the data matrix with k rows after applying the IQR criteria where k<=m
"""
def Perform_CAT12_QC(dataMatrix, data_dir, dataset, IQR_criteria, Features, sav_loc=False):
    report_files = glob.glob(os.path.join(data_dir + "report/", "catlog_*.txt"))
    IQR =[]
    ids = []
    substr = "Image Quality Rating (IQR):"
    QC_dir = dataset+"_CAT12_QC/"
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

    if (sav_loc==True):
        # check if the specified dir exists, otherwise create it
        if (os.path.exists(os.path.join(data_dir, QC_dir))==False):
            os.mkdir(os.path.join(data_dir, QC_dir))
        # convert to numpy array and save
        size = np.shape(X_CAT12_QC)
        np_CAT12_QC = X_CAT12_QC.to_numpy()
        np.save(data_dir+ QC_dir+"Prep_CAT12_" + Features + "_"+dataset+"_QC_"+str(size[0])+"x"+str(size[1]-2)+".npy", np_CAT12_QC  )
        X_CAT12_QC.to_csv(data_dir+ QC_dir+"Prep_CAT12_" + Features +"_"+dataset+"_QC_"+str(size[0])+"x"+str(size[1]-2)+".csv")
    return X_CAT12_QC


"""
Input: this function takes an IXI data or feature matrix and the metadata table 
-> Both tables must have original (raw) IDs at the zeroth index
Output: returns the data table with appended age and gender (if any) labels or information 
"""
def Get_IXI_Labels(IXI_Data_Matrix, IXI_Metadata):
    # extract the subject IDs from the metadata table
    # IXI_Metadata = Extract_IDs(IXI_Metadata, "metadata")

    # extract the subject IDs from the data matrix
    IXI_Data_Matrix = Extract_IDs(IXI_Data_Matrix, "feat")

    # rename the column names in both tables based on index 
    IXI_Metadata = IXI_Metadata.rename(columns={IXI_Metadata.columns[-1]: "ID"})
    IXI_Data_Matrix = IXI_Data_Matrix.rename(columns={IXI_Data_Matrix.columns[-1]: "ID"})

    # join the two tables based having same IDs
    IXI_Data_Matrix = IXI_Data_Matrix.merge(IXI_Metadata, how="inner", left_on="ID", right_on="IXI_ID" )


##### this function takes an IXI table (metadata or data matrix)   ######
##### and adds a new column with the updated ids    ###### 
def Extract_IDs(X, X_type):
    ids_new = []
    if (X_type=="metadata"):
        sub_id = "IXI_ID"
    elif(X_type=="feat"):
        sub_id = "names"
    ids_old = X.loc[:,sub_id]
    for i in range(0,len(ids_old)):
        id_i = re.search(r'IXI(.*?)-', ids_old[i])
        id_i =  int(id_i.groups(1)[0])
        ids_new.append(id_i)
    return X.append(ids_new)

"""
Input: this function takes an BraTS data or feature matrix and the metadata table 
-> Both tables must have original IDs at the zeroth index
Output: returns the data table with appended age and gender (if any) labels or information 
"""
def Get_BraTS_Labels(BraTS_Data_Matrix, BraTS_Metadata):
    # updates the metadata IDs 
    for i in range(0,len(BraTS_Metadata)):
        BraTS_Metadata.iloc[i,0] = str(BraTS_Metadata.iloc[i,0] + "_t1")
    
    merged = BraTS_Data_Matrix.merge(BraTS_Metadata, left_on="names", right_on="Brats20ID", how="inner", suffixes=("_tableA", "_tableB"))
    return merged.drop(columns=["Brats20ID",  "Extent_of_Resection"])

"""
Input: this function takes an ICBM data table and the metadata table 
-> Both tables must have original (raw) IDs at the zeroth index
Output: returns the data table with appended age and gender (if any) labels or information 
"""
def Get_ICBM_Labels(ICBM_Data_Matrix, ICBM_Metadata):
    # updates the data table IDs 
    dataMatrixIDs = ICBM_Data_Matrix.loc[:,"names"]
    for i in range(0,len(dataMatrixIDs)):
        id_i = re.search(r'ICBM_(.*?)_MR', dataMatrixIDs[i])
        if id_i is not None:   
            id_i =  id_i.groups(1)[0]
            ICBM_Data_Matrix.loc[i,"names"] = id_i

    merged = ICBM_Data_Matrix.merge(ICBM_Metadata, left_on="names", right_on="Subject_ID", how="inner")
    return merged.drop(columns=["Subject_ID", "Image_ID", "Description"])

def Get_ADNI_Labels(Data_Matrix, Metadata):
    # updates the metadata IDs 
    for i in range(0,len(Metadata)):
        Metadata.loc[i,"Subject_ID"] = "ADNI_" + str(Metadata.iloc[i,0])  + "_T1w"
    
    merged = Data_Matrix.merge(Metadata, left_on="names", right_on="Subject_ID", how="inner", suffixes=("_tableA", "_tableB"))
    return merged.drop(columns=["Subject_ID", "Protocol"])

"""
This function takes a metadata file and replaces it with the desired pattern
"""
def replaceLabels(metadata, dataset):
    for i in range(len(metadata)):
        if (dataset == "IXI"):
            id_i = re.search(r'IXI(.*?)-', metadata.loc[i,"Subject_ID"])
        elif (dataset == "ADNI"):
            id_i = re.search(r'ADNI_(.*)', metadata.loc[i,"Subject_ID"])
        if id_i is None:
            continue
        else:
            id_i =  id_i.groups(1)[0]
            if (dataset=="IXI"):
                metadata.loc[i,"Subject_ID"] = "IXI_"+id_i
            elif (dataset=="ADNI"):
                metadata.loc[i,"Subject_ID"] = "ADNI_"+id_i
    return metadata


"""
This function takes the feature or data matrix and the path as input and saves it to the desired local directory
"""
def Save_To_Local(DataMatrix, dataset, path, Features):
    size = np.shape(DataMatrix)
    np_DataMatrix = DataMatrix.to_numpy()
    if (Features=="thickness"):
        path = path +  dataset + "_SBM_thickness/"
        np.save(path + "Prep_CAT12_Thick_"+dataset+"_"+str(size[0])+"x"+str(size[1]-2)+".npy", np_DataMatrix )
        DataMatrix.to_csv(path + "Prep_CAT12_Thick_"+dataset+"_"+str(size[0])+"x"+str(size[1]-2)+".csv")
    elif (Features=="volume"):
        path = path +  dataset + "_VBM_volume/"
        np.save(path +  "Prep_CAT12_Vol_"+dataset+"_"+str(size[0])+"x"+str(size[1]-2)+".npy", np_DataMatrix )
        DataMatrix.to_csv(path + "Prep_CAT12_Vol_"+dataset+"_"+str(size[0])+"x"+str(size[1]-2)+".csv")
    elif (Features=="combined"):
        path = path +  dataset + "_VBM_volume/"
        np.save(path + "Prep_CAT12_Thick_Vol_"+dataset+"_"+str(size[0])+"x"+str(size[1]-2)+".npy", np_DataMatrix )
        DataMatrix.to_csv(path + "Prep_CAT12_Thick_Vol_"+dataset+"_"+str(size[0])+"x"+str(size[1]-2)+".csv")


"""
================================================================================================================
This function takes ADNI metadata table and returns distict rows with subjects having the desired modality protocols
"""
def Clean_ADNI_Metadata(Metadata, Modality):
    MRI_protocol = ["MPRAGE", "MPRAGE", "MP-RAGE" , "MP RAGE", "MPRAGE SAG", "MP RAGE SAGITTAL", "MPRAGE SAGITTAL", "ADNI       MPRAGE", "MP- RAGE"
    "ADNI_new   MPRAGE", "MP-RAGE-", "mprage","           MPRAGE"]
    fMRI_Protocol = ["Axial rsfMRI (Eyes Open)", "Axial MB rsfMRI (Eyes Open)", "Axial_rsFMRI_Eyes_Open", 
                    "Extended Resting State fMRI"]

    if (Modality=="MRI"):
        # search for MPRAGE protocol having different description
        ADNI_T1w_MRI = Metadata[Metadata["Description"].isin(MRI_protocol)]
        return ADNI_T1w_MRI.drop_duplicates(subset=["Subject"])

    elif (Modality=="rsfMRI"):
        # search for MPRAGE protocol having different description
        ADNI_rsfMRI = Metadata[Metadata["Description"].isin(fMRI_Protocol)]
        return ADNI_rsfMRI.drop_duplicates(subset=["Subject"])

    elif (Modality=="Multimodal"):
        # create three different tables, each containing demographics of participants with a single modality
        ADNI_MRI = Metadata.loc[Metadata["Modality"]=="MRI",:].drop_duplicates("Subject")
        ADNI_PET = Metadata.loc[Metadata["Modality"]=="PET",:].drop_duplicates("Subject")
        ADNI_fMRI = Metadata.loc[Metadata["Modality"]=="fMRI",:].drop_duplicates("Subject")

        # keep the desired MRI protocols (with Description column label) only
        ADNI_MRI = ADNI_MRI[ADNI_MRI["Description"].isin(MRI_protocol)]
        ADNI_fMRI = ADNI_fMRI[ADNI_fMRI["Description"].isin(fMRI_Protocol)]

        # join the three tables with Subject ids as primary and foriegn key
        # returns only the subjects who have records of all three mdalities
        print(ADNI_MRI.shape, ADNI_fMRI.shape, ADNI_PET.shape)
        ADNI_Multi = ADNI_MRI.merge(ADNI_PET, left_on="Subject", right_on="Subject",how="inner" ).merge(ADNI_fMRI, left_on="Subject", right_on="Subject",how="inner" )
        print(ADNI_Multi.shape)
        return ADNI_Multi
    
