#!/bin/bash

# this script performs parallel preprocessing with freesurfer recon-all 
# executes 8 jobs simulataneouly taking approx. 10 hours for full execution

ls .. | grep ^sub- > subjList.txt

for sub in `cat subjList.txt`; do
cp ../${sub}/ses-BL/anat/*.gz .
done

gunzip *.gz

SUBJECTS_DIR=`pwd`

ls *.nii | parallel --jobs 8 recon-all -s {.} -i {} -all -qcache

rm *.nii

for sub in `cat subjList.txt`; do
mv ${sub}_ses-BL_T1w.nii ${sub}
done
