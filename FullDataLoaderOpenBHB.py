import numpy as np
import pandas as pd
import os
import io
import glob
import re
import nilearn as nl
from nilearn.masking import apply_mask
from nilearn.image import index_img
from nilearn.input_data import NiftiMasker


fpaths_quasi = glob.glob(os.path.join(path_quasi, "*quasiraw_T1w.npy"))
fpaths_vbm = glob.glob(os.path.join(path_vbm, "*cat12vbm_desc-gm_T1w.npy"))
fpaths_fsl = glob.glob(os.path.join(path_fsl, "*freesurfer_desc-xhemi_T1w.npy"))
fpaths_vbm_roi = glob.glob(os.path.join(path_roi, "*cat12vbm_desc-gm_ROI.npy"))
fpaths_deskn_roi = glob.glob(os.path.join(path_roi, "*desikan_ROI.npy"))
fpaths_destrx_roi = glob.glob(os.path.join(path_roi, "*destrieux_ROI.npy"))

fpaths_vbm_roi = pd.DataFrame(fpaths_vbm_roi).sort_values(by=[0])
fpaths_deskn_roi = pd.DataFrame(fpaths_deskn_roi).sort_values(by=[0])
fpaths_destrx_roi = pd.DataFrame(fpaths_destrx_roi).sort_values(by=[0])
fpaths_vbm = pd.DataFrame(fpaths_vbm).sort_values(by=[0])
fpaths_fsl = pd.DataFrame(fpaths_fsl).sort_values(by=[0])
fpaths_quasi = pd.DataFrame(fpaths_quasi).sort_values(by=[0])


# the filenames in the training set don't contain the exactly matching unique IDs
# as in the metadata file, therefore last 6 digits in the ID need to be rounded to zeros
# rename the file names (round last 6 digits)
def renameFiles(dir, modality):
    if (modality =="vbm_gm"):
        fpaths = glob.glob(os.path.join(path_vbm, "*cat12vbm_desc-gm_T1w.npy"))
        str_pattern = "_preproc-cat12vbm_desc-gm_T1w"
        rename_mri(fpaths, str_pattern)
    elif (modality == "quasi_raw"):
        fpaths = glob.glob(os.path.join(path_quasi, "*quasiraw_T1w.npy"))
        str_pattern = "_preproc-quasiraw_T1w.npy"
        rename_mri(fpaths, str_pattern)

def rename_mri(fpaths, str_pattern)    
    for idx, val in enumerate(fpaths):
        id_i = re.search("sub-(.*)_prep", str(val))
        id_i = round(int(id_i.group(1)), -6)
        new_file = path_quasi + "sub-" + str(id_i) + str_pattern
        old_file = val
        os.rename(old_file , new_file )

"""
Returns the specified mask (img) file 
"""
def getMask(modality):
    MASKS = {
        "vbm": {
            "basename": "cat12vbm_space-MNI152_desc-gm_TPM.nii",
            "thr": 0.05},
        "quasiraw": {
            "basename": "quasiraw_space-MNI152_desc-brain_T1w.nii",
            "thr": 0}
    }

    img = nib.load('/media/dataanalyticlab/Drive2/MANSOOR/Dataset/OpenBHB/resource/cat12vbm_space-MNI152_desc-gm_TPM.nii')

    mask_vbm = np.array(img.dataobj)

    img_quasi_mask = nib.load('/media/dataanalyticlab/Drive2/MANSOOR/Dataset/OpenBHB/resource/quasiraw_space-MNI152_desc-brain_T1w.nii').get_fdata()
    # im = nibabel.Nifti1Image(arr.squeeze(), affine)
    # arr = apply_mask(im, masks[key])

    # mask = np.array(img_quasi_mask.dataobj)

    # retuns a 2D array with ones on the diagonals and zeros on non-diagnonal enteries
    affine = np.eye(4)
    resourcedir = '/media/dataanalyticlab/Drive2/MANSOOR/Dataset/OpenBHB/resource/'
    masks = dict((key, os.path.join(resourcedir, val["basename"]))
                    for key, val in MASKS.items())
    for key in masks:
        arr = nib.load(masks[key]).get_fdata()
        thr = MASKS[key]["thr"]
        arr[arr <= thr] = 0
        arr[arr > thr] = 1
        masks[key] = nib.Nifti1Image(arr.astype(int), affine)

    arr = index_img('/media/dataanalyticlab/Drive2/MANSOOR/Dataset/OpenBHB/resource/cat12vbm_space-MNI152_desc-gm_TPM.nii',0).get_fdata()
    thr = MASKS["vbm"]["thr"] #### thr = 0.05
    arr[arr <= thr] = 0
    arr[arr > thr] = 1
    mask_vbm = nib.Nifti1Image(arr.astype(int), affine)


""""
This function loads all data (.npy) modalities (roi, vbm, sbm, quasi-raw)
Flattens the array and merges all modalities together in an 2d matrix
Saves the merged 2D matrix (.npy) locally
"""
def loadDumpFullData():
    sub_metadata = pd.DataFrame(dtype=float)
    for i in range(0,3209):
        for j in range(0,3):
            if j == 0:
                vbm_roi_i = np.load(all_paths[i,j]).flatten().reshape(1,284)
            if j ==1:
                deskn_roi_i = np.load(all_paths[i,j]).flatten().reshape(1,476)
            if j == 2:
                destrx_roi_i = np.load(all_paths[i,j]).flatten().reshape(1,1036)
            if j==3:
                # masking of the CAT12 VBM
                unmasked_vbm_i = np.load(all_paths[i,j], mmap_mode="r")
                img = nib.Nifti1Image(unmasked_vbm_i.squeeze(), affine)
                masked_vbm_i = apply_mask(img, mask_vbm)
                masked_vbm_i = np.expand_dims(masked_vbm_i, axis=0)
            if j==4:
                fsl_xhemi_i = np.load(all_paths[i,j]).flatten().reshape(1,1310736)
            if j==5:
                # masking of the QuasiRaw MRI
                unmasked_quasi_i = np.load(all_paths[i,j], mmap_mode="r")
                im = nib.Nifti1Image(unmasked_quasi_i.squeeze(), affine)
                masked_quasi_i = apply_mask(im, masks['quasiraw'])
                masked_quasi_i = np.expand_dims(masked_quasi_i, axis=0)
        
        # extract subject unique IDs from the file_name
        # round-off last 6 digits in the ID and insert zeros inplace 
        id_i = re.search("sub-(.*)_prep", all_paths[i,j])
        id_i = round(int(id_i.group(1)), -6)
        
        age = participants["age"].loc[participants['participant_id'] == id_i]
        gender = participants["sex"].loc[participants['participant_id'] == id_i]
        age = age.astype(float)
        # sub_metadata = sub_metadata.append([[id_i,age.values, gender.values]], ignore_index=True)
        concat_row_i = np.concatenate((np.array(id_i).reshape(1,1), vbm_roi_i, deskn_roi_i,
                destrx_roi_i, 
                np.array(gender.values).reshape(1,1), np.array(age.values).reshape(1,1)) , axis=1)
        # print("Total size of a full row (1,3659575): ", concat_row_i.nbytes, "bytes")
        with open ("/media/dataanalyticlab/Drive2/MANSOOR/Dataset/OpenBHB/openbhb_train_roi_dataset.npy",'a') as f_object:
            writer_object = writer(f_object, delimiter=",")
            writer_object.writerow(concat_row_i.reshape(1799))
            f_object.close()
        
    # sub_metadata.to_csv('/media/dataanalyticlab/Drive2/MANSOOR/sub_info_test.csv',index=False)  