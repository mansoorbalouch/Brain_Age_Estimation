%{
Run the preprocessing pipeline on the structural MRI using parallel
processing (parpool) in matlab.
Input: provide the spm12 installation path and the directory where the raw
nifti MRI files are present.
Output: 

%}

% Path to SPM12 installation
% spm_path = 'C:\Program Files\MATLAB\spm12';
spm_path = '/media/dataanalyticlab/Drive2/MATLAB/spm12';

% Input and output directories
% input_dir = 'C:\Users\mansoor.ahmed\Documents\Mansoor\MRI_Preprocess_SPM_Test';
% output_dir = 'C:\Users\mansoor.ahmed\Documents\Mansoor\MRI_Preprocess_SPM_Test\Output';

input_dir = '/media/dataanalyticlab/Drive2/MANSOOR/Dataset/Test/MRI_Preprocess_SPM_Test';
output_dir = '/media/dataanalyticlab/Drive2/MANSOOR/Dataset/Test/MRI_Preprocess_SPM_Test/Output';

% Initialize SPM
addpath(spm_path);
spm('defaults', 'PET');

% Get a list of subject folders or files
subject_list = dir(fullfile(input_dir, '*.nii'));
num_subjects = length(subject_list);


% Start parallel pool (optional but recommended for efficiency)
num_workers = 8;
pool = parpool(num_workers);  % Start parallel pool using all available CPU cores

% Loop through subjects using parfor
parfor subject_idx = 1:num_subjects
    subject_file = fullfile(subject_list(subject_idx).folder, subject_list(subject_idx).name);
    fprintf(subject_file)
%     processSubject(subject_file, output_dir);
end

% Delete parallel pool when done
delete(pool);

fprintf('All subjects processed.\n');
