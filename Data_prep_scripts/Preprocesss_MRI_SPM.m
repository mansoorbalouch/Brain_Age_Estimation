% Path to SPM12 installation
% spm_path = 'C:\Program Files\MATLAB\spm12';

% Input and output directories
% input_dir = 'C:\Users\mansoor.ahmed\Documents\Mansoor\MRI_Preprocess_SPM_Test';
% output_dir = 'C:\Users\mansoor.ahmed\Documents\Mansoor\MRI_Preprocess_SPM_Test\Output';

spm_path = '/media/dataanalyticlab/Drive2/MATLAB/spm12';
input_dir = '/media/dataanalyticlab/Drive2/MANSOOR/Dataset/Test/MRI_Preprocess_SPM_Test';
output_dir = '/media/dataanalyticlab/Drive2/MANSOOR/Dataset/Test/MRI_Preprocess_SPM_Test/Output';


% Initialize SPM
addpath(spm_path);
spm('defaults', 'PET');

% Get a list of subject folders or files
subject_list = dir(fullfile(input_dir, '*.nii'));
num_subjects = length(subject_list);

% Preallocate arrays for quality parameters, region-wise volumes, and other features
quality_params = zeros(num_subjects, 2);
region_volumes = zeros(num_subjects, 2); % 1st column for GM, 2nd column for WM

% Loop through subjects
for subject_idx = 1:num_subjects
    subject_file = fullfile(subject_list(subject_idx).folder, subject_list(subject_idx).name);
    processSubject(subject_file, output_dir, input_dir, spm_path);

end

fprintf('All subjects processed.\n');
