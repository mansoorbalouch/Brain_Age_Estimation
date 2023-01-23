import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
from math import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler


train_path = "/media/dataanalyticlab/Drive2/MANSOOR/row_100.csv"
val_path = "/media/dataanalyticlab/Drive2/MANSOOR/row_100.csv"
sub_labels_path = "/media/dataanalyticlab/Drive2/MANSOOR/Dataset/OpenBHB/Subjects/All_subjects_metadata.csv"

X_train = np.loadtxt(open(train_path), delimiter=",", usecols=np.arange(1,3659573))
X_val = np.loadtxt(open(val_path), delimiter=",", usecols=np.arange(1,3659573))
sub_labels = np.genfromtxt(open(sub_labels_path), delimiter=",", usecols=(0,1), skip_header=True, names=None, skip_footer=True)
Y = sub_labels[:,1]


X = np.concatenate((X_train, X_val))

scaler = MinMaxScaler()
X_sc = scaler.fit_transform(X)

print("Data matrix min-max scaled...")


# compute covariance between x and y
def comp_cov(x, y, m1, m2):
    return sum([(xi - m1) * (yi - m2) for xi, yi in zip(x, y)]) / (len(x))

# compute entropy
def entropy(values):
    values = values.astype(int)
    counts = np.bincount(values)
    probs = counts[np.nonzero(counts)] / float(len(values))
    return - np.sum(probs * np.log(probs))


Index = []
mean_x = []
std_x = []
median = []
min_x = []
max_x = []
var_x = []
cov_xy = []
corr_xy = []
entropy_x = []
MI = []
NMI_score = []

mu_y = np.mean(Y[:,])
std_y = np.std(Y[:,])
entropy_y = entropy(Y[:,])



# Compute the feature selection metrics 
for i in range(0, len(X_sc.transpose())):
    Index.append(i)         # feature index
    mu_x = np.mean(X_sc[:,i])  # compute mean (x_i)
    mean_x.append(mu_x)
    st_x = np.std(X_sc[:,i])  # compute std(x_i)
    std_x.append(st_x)
    min_x.append(np.min(X_sc[:,i]))  # find min(x_i)
    max_x.append(np.max(X_sc[:,i]))   # find max (x_i)
    var_x.append(comp_cov(X_sc[:,i], X_sc[:,i], mu_x, mu_x))  # compute var(x)
    cov = comp_cov(X_sc[:,i], Y[:,], mu_x, mu_y)   # compute covariance(x_i, y)
    cov_xy.append(cov)
    if st_x != 0:
        corr_xy.append(cov/(st_x * std_y))       # compute correlation(x_i, y)
    else:
        corr_xy.append(0)
    # Entropy of feature x_i
    entropy_x.append(entropy(X_sc[:,i]))
    # Mutual Information computing the statistical dependence b/w X and Y
    MI.append(mutual_info_regression(X_sc[:,i].reshape(-1,1),Y[:,]))
    # Normalized Mutual Information Score (b/w 0 and 1)
    # NMI = MI(X,Y)/mean{H(X), H(Y)}
    NMI_score.append(normalized_mutual_info_score(X_sc[:,i], Y[:,]))


feature_metrics = np.array((Index, mean_x, std_x, min_x, max_x, var_x, cov_xy, corr_xy, entropy_x, MI, NMI_score)).transpose()

### save the feature metrics
feat_path = "/media/dataanalyticlab/Drive2/MANSOOR/"
np.savetxt(feat_path+"Feature_metrics_openbhb.csv", feature_metrics, delimiter=',',
header="Id, Mean(x), Std(x), Min(x), Max(x), Var(x), Cov(xy), Corr(xy), Entropy(x), MI(xy), NMI(xy)")

