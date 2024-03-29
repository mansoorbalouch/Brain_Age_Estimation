# Brain Age Estimation
Brain age estimation refers to the prediction of the biological brain age (apparently how old a person's brain looks like) using the MRIs. Brain estimated age difference, Brain-EAD, is the difference between a person's biological brain age and the chronological age. Brain-EAD is a data-driven biomarker for different neurological disorders and employs machine learning techniques on the structural and functional brain MRIs of healthy controls (HC) for developing such a framework. A detailed neuroimaging pipeline is followed for pre-processing the MRIs and, subsequently, features extraction using tools such as FreeSurfer, FSL, SPM12, and CAT12. Similarly, the data science approaches such as feature selection, dimensionality reduction, statistical tests, and data visualization are widely applied for developing efficient brain age estimation frameworks.
A series of steps are executed for this purpose:
## T1-weighted MRI Preprocessing 
The 3D MRIs of different subjects (or same subject at different time-points in longitudinal studies) are preprocessed following the standard pipepline of Voxel Based Morphometry (VBM) or Surface Based Morphometry (SBM) as explained in the [research article](https://doi.org/10.1038/s42003-022-03880-1). Briefly, the standardized VBM workflow includes the segmentation of gray matter (GM), white matter (WM), and cerebrospinal fluid (CSF) from other brain tissues and skull, spatial normalization into standard space such as MNI152 or Taliarch space, and smoothing with a Gaussian kernel (of thickness 2mm, 4mm, or even 6mm) before inferential statistics are applied. Later, the preprocessed MRIs are commonly used for analysing (1) group differences in terms of regional GM volume (GMV), WMV, etc. between patients and controls or men and women or (2) associations between individual structural and functional variations in regional GMV and behavioral phenotypes, including learning, age, or disorder-relevant traits. Neuroimaging tools such as SPM12, CAT12, and FreeSurfer are used for the automated preprocessing of MRI and PET. 
## Feature Extraction
The aim of this step is to transform a very high dimensional data matrix, i.e., where the number of features or dimensions or columns ***n*** are significantly higher than the number of data samples or points ***m*** (***m<<n***). Broadly, two approaches are applied for projecting the data from a high-dimensional space ***mxn*** to a lower dimensional space ***mxk*** by retaining the variation in the data. These are known as linear and non-linear dimensionality reduction approaches. For brain age estimation studies, Priniciple Component Analysis (PCA) is being widely used when the voxel-based features (of the MRIs) have very high dimensionality and effect the ultimate analysis such as regression.
## Feature Engineering and Feature Selection
Proper feature selection has two main ways, filter, and wrapper feature selection. The criterion for filter feature selection is independent of the particular classification algorithm, generally, estimation of statistical properties are performed using them. The 6 most common criteria are the Kruskal-Wallis test, Chi-square score, multivariate minimal redundancy maximal relevance (MRMR), Fisher score, Gini score, and relief feature score.
## Exploratory Data Analysis

## Model training
Firstly,the data is split into train-test set following ***80:20*** division. The training set is further the brain age estimation model is trained on t 

## Performance Evaluation and Results Visualization

## Dependencies Installation
- pandas
- numpy
- scikit-learn
- matplotlib
- nibabel
- nilearn
- nipype
- keras
- tensorflow
- scikeras

