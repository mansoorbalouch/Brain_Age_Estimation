function processSubject(subject_file, output_dir,input_dir, spm_path)
    [~, subject_name, ~] = fileparts(subject_file);
    output_subject_dir = fullfile(output_dir, subject_name);
    mkdir(output_subject_dir);

    % Voxel size for resampling (use AFNI's dicom_hdr on the functional & structural DICOM files and use the "slice thickness")
    fVoxSize = [3 3 3]; % functionals (non-multiband usually [3 3 3], multiband usually [2 2 2])
    sVoxSize = [1 1 1]; % for structID{1} (MPRAGE usually [1 1 1], MBW usually [3 3 3])
    
    % smoothing kernel for functionals  (mm isotropic)
    FWHM = 8;

    fprintf('Processing subject: %s\n', subject_name);

    % Preprocessing steps
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {subject_file};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
    % write save the results immediately after each preprocessing steps and takes two flags:
    % first tells it whether to save the preprocessed image and second tell
    % it to save the unpreprocessed file
    matlabbatch{1}.spm.spatial.preproc.channel.write = [0, 0];

    matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {fullfile(spm_path, 'tpm', 'TPM.nii,1')};
    matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1; 
    matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];  
    matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0]; 
    % Specify the output path for c1 (gray matter) file
    matlabbatch{1}.spm.spatial.preproc.tissue(1).write = [0,0];

    matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {fullfile(spm('dir'), 'tpm', 'TPM.nii,2')};
    matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1; 
    matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 0]; 
    matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0]; 
    matlabbatch{1}.spm.spatial.preproc.tissue(2).write = [0,0];
    
    matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {fullfile(spm('dir'), 'tpm', 'TPM.nii,3')};
    matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2; 
    matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0]; 
    matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0]; 
    matlabbatch{1}.spm.spatial.preproc.tissue(3).write = [0,0];

    fprintf('Tissue segmentation completed ..\n');

    matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1; 
    matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1; 
    matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2]; 
    matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni'; 
    matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0; 
    matlabbatch{1}.spm.spatial.preproc.warp.samp = 3; 
    matlabbatch{1}.spm.spatial.preproc.warp.write = [0, 0]; 

    fprintf('MRI warping completed..\n');

    % Module for normalization 
    matlabbatch{1}.spm.tools.dartel.warp.settings.template = 'Template';
    matlabbatch{1}.spm.tools.dartel.warp.settings.rform = 0;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).rparam = [4 2 1e-06];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).K = 0;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).slam = 16;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).rparam = [2 1 1e-06];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).K = 0;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).slam = 8;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).rparam = [1 0.5 1e-06];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).K = 1;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).slam = 4;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).rparam = [0.5 0.25 1e-06];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).K = 2;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).slam = 2;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).rparam = [0.15 0.125 1e-06];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).K = 4;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).slam = 1;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).rparam = [0.25 0.125 1e-06];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).K = 6;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).slam = 0.5;
    matlabbatch{1}.spm.tools.dartel.warp.settings.optim.lmreg = 0.01;
    matlabbatch{1}.spm.tools.dartel.warp.settings.optim.cyc = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.optim.its = 3;

% 
    % DARTEL: normalize & smooth functional images to MNI
    matlabbatch{1}.spm.tools.dartel.mni_norm.template(1) = cfg_dep('Run Dartel (create Templates): Template (Iteration 6)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','template', '()',{7}));                                        
    matlabbatch{1}.spm.tools.dartel.mni_norm.vox = fVoxSize;
    matlabbatch{1}.spm.tools.dartel.mni_norm.bb = [NaN NaN NaN; NaN NaN NaN];
    matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 0;
    matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [FWHM FWHM FWHM];
    
    % DARTEL: normalize bias-corrected MPRAGE to MNI
    matlabbatch{1}.spm.tools.dartel.mni_norm.template(1) = cfg_dep('Run Dartel (create Templates): Template (Iteration 6)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','template', '()',{7}));
    matlabbatch{1}.spm.tools.dartel.mni_norm.vox = sVoxSize;
    matlabbatch{1}.spm.tools.dartel.mni_norm.bb = [NaN NaN NaN; NaN NaN NaN];
    matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 0;
    matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [0 0 0];
    
    % DARTEL: normalize grey matter segmentation to MNI (for Conn toolbox or other use of segmentations)
    matlabbatch{1}.spm.tools.dartel.mni_norm.template(1) = cfg_dep('Run Dartel (create Templates): Template (Iteration 6)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','template', '()',{7}));
    matlabbatch{1}.spm.tools.dartel.mni_norm.vox = sVoxSize;
    matlabbatch{1}.spm.tools.dartel.mni_norm.bb = [NaN NaN NaN; NaN NaN NaN];
    matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 0;
    matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [0 0 0];
    
    % DARTEL: normalize white matter segmentation to MNI (for Conn toolbox or other use of segmentations)
    matlabbatch{1}.spm.tools.dartel.mni_norm.template(1) = cfg_dep('Run Dartel (create Templates): Template (Iteration 6)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','template', '()',{7}));
    matlabbatch{1}.spm.tools.dartel.mni_norm.vox = sVoxSize;
    matlabbatch{1}.spm.tools.dartel.mni_norm.bb = [NaN NaN NaN; NaN NaN NaN];
    matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 0;
    matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [0 0 0];
    
    % DARTEL: normalize CSF segmentation to MNI (for Conn toolbox or other use of segmentations)
    matlabbatch{1}.spm.tools.dartel.mni_norm.template(1) = cfg_dep('Run Dartel (create Templates): Template (Iteration 6)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','template', '()',{7}));
    matlabbatch{1}.spm.tools.dartel.mni_norm.vox = sVoxSize;
    matlabbatch{1}.spm.tools.dartel.mni_norm.bb = [NaN NaN NaN; NaN NaN NaN];
    matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 0;
    matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [0 0 0];
    
    % DARTEL: normalize functional images to MNI no smoothing
    matlabbatch{1}.spm.tools.dartel.mni_norm.template(1) = cfg_dep('Run Dartel (create Templates): Template (Iteration 6)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','template', '()',{7}));
    matlabbatch{1}.spm.tools.dartel.mni_norm.vox = fVoxSize;
    matlabbatch{1}.spm.tools.dartel.mni_norm.bb = [NaN NaN NaN; NaN NaN NaN];
    matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 0;
    matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [0 0 0];



%     ---------------------------------------------------------

%     matlabbatch{1}.spm.spatial.normalise.write.subj.def = fullfile(output_subject_dir, 'y_T1.nii'); 
%     matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {fullfile(output_subject_dir, 'c1T1.nii'), fullfile(output_subject_dir, 'c2T1.nii'), fullfile(output_subject_dir, 'c3T1.nii')}; 
%     matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70; 78 76 85]; 
%     matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [1 1 1]; 
%     matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 1; 
%     fprintf('Tissue normalized..\n');
%     matlabbatch{1}.spm.spatial.smooth.data = fullfile(output_subject_dir, 'wc1T1.nii');
%     matlabbatch{1}.spm.spatial.smooth.fwhm = [8 8 8]; 
%     matlabbatch{1}.spm.spatial.smooth.dtype = 0; 
%     matlabbatch{1}.spm.spatial.smooth.im = 0; 
%     matlabbatch{1}.spm.spatial.smooth.prefix = 's'; 
%     matlabbatch{1}.spm.spatial.smooth.write = [0, 0]; 

  
%   Run the preprocessing matlabbatch
    spm_jobman('run', matlabbatch);
    fprintf('matlabbatch run completed..\n');

    % Save the preprocessed raw MRI
    preprocessed_mri_output = fullfile(output_subject_dir, 'preprocessed_T1.nii');
    copyfile(subject_file, preprocessed_mri_output);

    gm_image = fullfile(input_dir, strcat('c1',subject_name,'.nii'));
    wm_image = fullfile(input_dir, strcat('c2',subject_name,'.nii'));
    csf_image = fullfile(input_dir, strcat('c3',subject_name,'.nii'));
    seg8_file = fullfile(input_dir, strcat(subject_name,'_seg8.mat'));

    % Save the segmented GM and WM images
    movefile(gm_image, output_subject_dir);
    movefile(wm_image, output_subject_dir);
    movefile(csf_image, output_subject_dir);
    movefile(seg8_file, output_subject_dir);

    fprintf("Segmented output files saved..\n");
    fprintf('Analysis complete for subject: %s\n', subject_name);
end