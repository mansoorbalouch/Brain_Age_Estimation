#!/bin/bash

echo "MRI processing step(s) to perform? vbm_preprocess or smooth!"
read step

cat_std_dir=/media/dataanalyticlab/Drive2/MATLAB/spm12/toolbox/cat12/standalone/
cat_std_file=${cat_std_dir}cat_standalone.sh
matlab_path=/media/dataanalyticlab/Drive2/MATLAB/R2021b/bin/matlab
cat_vbm_prep_pipeline=${cat_std_dir}cat_standalone_segment.m
cat_smooth=${cat_std_dir}cat_standalone_smooth.m
cat_parallelize=${cat_std_dir}cat_parallelize.sh

if [ $step = "vbm_preprocess" ]; then
	echo "Insert the source nifti files path!"
	read  source_nii_files
	echo "MRI VBM preprocessing started..."
	$cat_std_file -ns -e -m $matlab_path -b $cat_vbm_prep_pipeline $source_nii_files
	echo "MRI VBM preprocessing finished..."
elif [ $step = "smooth" ]; then
	echo "Insert the preprocessed GM, WM or CSF nifti files (mwp[123]) path!"
	read  source_gm_nii_files
	echo "MRI smoothing started..."
	$cat_std_file -ns -e -m $matlab_path -b $cat_smooth $source_gm_nii_files
	echo "MRI smoothing finished..."	
fi;
