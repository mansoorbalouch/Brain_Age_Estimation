import numpy as np
import nibabel as nib
import nilearn as nl
from nilearn.masking import apply_mask
from nilearn.image import index_img
import re
from nilearn import masking
from nilearn.input_data import NiftiMasker
from nilearn import plotting
import dicom2nifti
import os
import glob
from nibabel import load, save, Nifti1Image
import warnings

warnings.filterwarnings('ignore')

############ convert ge to nifti and rename the files  #############
def Ge2Nii(sub_dir, out_dir, dataset):
    sub_dir_list = os.listdir(sub_dir)
    for subject in sub_dir_list:
        # create new folder for each subject if doesn't already exist
        sub_new_nii_dir = os.path.join(out_dir,subject)
        if (os.path.exists(sub_new_nii_dir) == False):
            os.mkdir(sub_new_nii_dir)
            # convert dicom to nifti here 
            temp_dir = os.listdir(os.path.join(sub_dir, subject))
            if len(temp_dir) == 0:       # Checking if the list is empty or not
                print("Empty directory, therefore can't convert!!")
            else:
                dicom2nifti.convert_ge.dicom_to_nifti(os.path.join(sub_dir, subject), os.path.join(out_dir,subject))
                print("MRI ge file successfully converted to nifti format...")
                # rename the converted nifti file with proper naming convention
                old_file = str(glob.glob((out_dir + subject + "/*.nii.gz"))[0])
                new_file = os.path.join(out_dir, subject , dataset+ "_"+subject+"_T1w.nii.gz")
                os.rename(old_file , new_file )


############ convert dicom to nifti and rename the files  ##################
def Dicom2Nii(sub_dir, out_dir, dataset):
    sub_dir_list = os.listdir(sub_dir)
    # loop through all the subjects in the dicom MRI data directory
    # list contains the subject IDs or the folder names
    for subject in sub_dir_list:
        # create new folder for each subject if doesn't already exist
        sub_new_nii_dir = os.path.join(out_dir,subject)
        if (os.path.exists(sub_new_nii_dir) == False):
            os.mkdir(sub_new_nii_dir)
            # convert dicom to nifti here 
            temp_dir = os.listdir(os.path.join(sub_dir, subject))
            if len(temp_dir) == 0:       # Checking if the list is empty or not
                print("Empty directory for subject " + subject +", therefore can't convert!!")
            else:
                dicom2nifti.convert_directory(os.path.join(sub_dir, subject), os.path.join(out_dir,subject))
                print("Converted dcm file of " + subject + " to nii format...")
                # rename the converted nifti file with proper naming convention
                old_file = str(glob.glob((out_dir + subject + "/*.nii.gz"))[0])
                new_file = os.path.join(out_dir, subject , dataset+ "_"+subject+"_T1w.nii.gz")
                os.rename(old_file , new_file )


################ convert mnc to nifti format ###########
def Mnc2Nii(sub_dir, out_dir):
    sub_dir_list = os.listdir(sub_dir)
    sub_dir_list = sub_dir_list[2:]

    for subject in sub_dir_list:
        minc = load(os.path.join(sub_dir, subject))
        basename = minc.get_filename().split(os.extsep, 1)[0]

        affine = np.array([[0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]])

        out = Nifti1Image(minc.get_data(), affine=affine)
        save(out, basename + '.nii')

############# Checks and deletes empty diretories ################
def Rem_Empty_Dir(input_dir):
    ADNI_MRI_dir_nii_list = os.listdir(input_dir)
    for subject in ADNI_MRI_dir_nii_list:
        sub_i_dir = os.path.join(sub_i_dir, subject)
        temp_dir = os.listdir(sub_i_dir)
        if len(temp_dir) == 0:       # Checking if the list is empty or not
            print("Empty directory for subject " + subject)
            # remove the empty directory of subject i
            os.rmdir(sub_i_dir)