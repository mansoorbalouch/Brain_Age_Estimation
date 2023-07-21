#!/bin/bash

dir="/media/dataanalyticlab/Drive2/MANSOOR/Dataset/ADNI/ADNI_CN_Only/ADNI/"
dst_dir="/media/dataanalyticlab/Drive2/MANSOOR/Dataset/ADNI/ADNI_CN_Only/CN_ADNI_MPRAGE/"
mp_dir1="MPRAGE"
mp_dir2="MP RAGE"
mp_dir3="MP-RAGE"
for sub in $dir*/; do # loop through all subjects in the directory
	if [ -d "$dir$(basename "$sub")/$mp_dir1" ]; then  # check for MPRAGE MRI protocol 
		echo "MP-RAGE directory exists for sub $(basename "$sub")..."; 
		mkdir $dst_dir$(basename "$sub");
		cd $dir$(basename "$sub")/$mp_dir1/;
		if [ $(ls | wc -l) -gt 1 ]; then   # check for multiple sessions of the same subject
			session_1="$(basename $dir$(basename "$sub")/$mp_dir1/*/)"
			cp $dir$(basename "$sub")/$mp_dir1/$session_1/*/*.dcm $dst_dir$(basename "$sub")/;
			echo "More than one sessions found for the subject $(basename "$sub"), copyied .."
		else 
			cp $dir$(basename "$sub")/$mp_dir1/*/*/*.dcm $dst_dir$(basename "$sub")/;
		fi;
	elif [ -d "$dir$(basename "$sub")/$mp_dir2" ]; then  # check for MPRAGE MRI protocol 
		echo "MP-RAGE directory exists for sub $(basename "$sub")..."; 
		mkdir $dst_dir$(basename "$sub");
		cd $dir$(basename "$sub")/$mp_dr2/;
		if [ $(ls | wc -l) -gt 1 ]; then   # check for multiple sessions of the same subject
			session_1="$(basename $dir$(basename "$sub")/$mp_dir2/*/)"
			cp $dir$(basename "$sub")/$mp_dir2/$session_1/*/*.dcm $dst_dir$(basename "$sub")/;
			echo "More than one sessions found for the subject $(basename "$sub"), copyied .."
		else 
			cp $dir$(basename "$sub")/$mp_dir2/*/*/*.dcm $dst_dir$(basename "$sub")/;
		fi;
	elif [ -d "$dir$(basename "$sub")/$mp_dir3" ]; then  # check for MPRAGE MRI protocol 
		echo "MP-RAGE directory exists for sub $(basename "$sub")..."; 
		mkdir $dst_dir$(basename "$sub");
		cd $dir$(basename "$sub")/$mp_dir3/;
		if [ $(ls | wc -l) -gt 1 ]; then     # check for multiple sessions of the same subject
			session_1="$(basename $dir$(basename "$sub")/$mp_dir3/*/)"
			cp $dir$(basename "$sub")/$mp_dir3/$session_1/*/*.dcm $dst_dir$(basename "$sub")/;
			echo "More than one sessions found for the subject $(basename "$sub"), copyied .."
		else 
			cp $dir$(basename "$sub")/$mp_dir3/*/*/*.dcm $dst_dir$(basename "$sub")/;
		fi;
	fi; 
	cd ..;
done
