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
from nibabel import load, save, Nifti1Image
import warnings

warnings.filterwarnings('ignore')

############ convert ge to nifti and rename the files  #############
def Ge2Nii(sub_dir, out_dir):
    sub_dir_list = os.listdir(sub_dir)
    for subject in sub_dir_list:
        os.mkdir(os.path.join(out_dir,subject))
        dicom2nifti.convert_ge.dicom_to_nifti(os.path.join(sub_dir, subject), os.path.join(out_dir,subject))
        # os.rename(os.path.join(out_dir,subject,"9_mprage_t1_ax_08_mm_ti-808.nii.gz"), os.path.join(out_dir,subject,"UTHC_"+subject+"_T1w.nii.gz"))


############ convert dicom to nifti and rename the files  ##################
def Dicom2Nii(sub_dir, out_dir):
    sub_dir_list = os.listdir(sub_dir)
    for subject in sub_dir_list:
        os.mkdir(os.path.join(out_dir,subject))
        dicom2nifti.convert_directory(os.path.join(sub_dir, subject), os.path.join(out_dir,subject))
        os.rename(os.path.join(out_dir,subject,"9_mprage_t1_ax_08_mm_ti-808.nii.gz"), os.path.join(out_dir,subject,"UTHC_"+subject+"_T1w.nii.gz"))


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