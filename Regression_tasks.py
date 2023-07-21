
import numpy as np
import pandas as pd
import os
import io
import glob
import math
import sklearn
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import LinearSVR
from sklearn_rvm import EMRVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, StratifiedShuffleSplit, KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import timeit
import warnings

warnings.filterwarnings('ignore')



dataset, Model, MAE, MSE , RMSE , R_sq , run_time, Set_type = ([] for i in range(8))

print("Package import done")

####################### Linear Regression ####################################
def LR(X_train, Y_train, X_test, Y_test, Feat):
    print("LR started running")
    start = timeit.default_timer()
    lm = LinearRegression()

    lm.fit(X_train, Y_train)

    # print("Intercept: {:,.3f}".format(lm.intercept_))
    # print("Coefficient: {:,.3f}".format(lm.coef_[1]))

    algo = "LR"
    Y_pred = lm.predict(X_test)
    
    stop = timeit.default_timer()
    time_new = stop - start

    results = Add_reg_results(Y_pred, Y_test, Feat, algo, time_new, "Test")
    print("LR finished running")
    return Y_pred, results, lm


################## Linear SVR  ###########################################
def LSVR(X_train, Y_train, X_test, Y_test, Feat):
    print("SVR started running")
    start = timeit.default_timer()
    eps = 5
    svr = LinearSVR(epsilon=eps, C=5, fit_intercept=True)
    svr.fit(X_train, Y_train)

    algo = "LSVR"
    Y_pred = svr.predict(X_test)

    stop = timeit.default_timer()
    time_new = stop - start

    results = Add_reg_results(Y_pred, Y_test, Feat, algo, time_new, "Test")
    print("SVR finished running")
    return Y_pred, results, svr


################## Relevance Vector Regression  ############################
def RVR(X_train, Y_train, X_test, Y_test, Feat):
    print("RVR started running")
    start = timeit.default_timer()
    pipe = Pipeline([
        ("scale", NanImputeScaler()),
        ("rvr", EMRVR(kernel="poly"))
    ]).fit(X_train, Y_train)

    algo = "RVR"
    Y_pred = pipe.predict(X_test)

    stop = timeit.default_timer()
    time_new = stop - start

    results = Add_reg_results(Y_pred, Y_test, Feat, algo, time_new, "Test")
    print("RVR finished running")
    return Y_pred, results, pipe


################ Generalized Linear Model ################################

#********* Gamma Distribution (positive skewed) ********#

def GLM_Gamma(X_train, Y_train, X_test, Y_test, Feat):
    print("GLM_Gamma started running")
    start = timeit.default_timer()
    glm_gamma = linear_model.GammaRegressor(alpha=8)
    glm_gamma.fit(X_train, Y_train)

    algo = "GLM_Gamma"
    Y_pred = glm_gamma.predict(X_test)

    stop = timeit.default_timer()
    time_new = stop - start

    results = Add_reg_results(Y_pred, Y_test, Feat, algo, time_new, "Test")
    print("GLM_Gamma finished running")
    return Y_pred, results, glm_gamma


#********* Tweedie Regressor (Normal Distribution) ********#

def GLM_normal(X_train, Y_train, X_test, Y_test, Feat):
    print("GLM_Normal started running")
    start = timeit.default_timer()

    glm = linear_model.TweedieRegressor()
    glm.fit(X_train, Y_train)
    algo = "GLM_Normal"
    Y_pred = glm.predict(X_test)
    stop = timeit.default_timer()
    time_new = stop - start

    results = Add_reg_results(Y_pred, Y_test, Feat, algo, time_new, "Test")
    print("GLM_Normal finished running")
    return Y_pred, results, glm


"""*************** Multi-Layer Perceptron (Regression) ******************
-> select the parameters using grid search 
"""
def MLP_Reg(X_train, Y_train, X_test, Y_test, Feat):
    print("MLP started running")
    start = timeit.default_timer()
    regr = MLPRegressor(random_state=1)

    parameters_MLP_reg = {
        'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
        'activation': ('identity',  'tanh', 'adam'),
        'alpha': (0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1),
        'beta_1': (0.1,0.9,0.1),
        'beta_2': (0.1,0.9,0.1),
        'learning_rate': ['constant','adaptive']
                    }
    # with GridSearch
    grid_search_MLP_reg = GridSearchCV(
        estimator=regr,
        param_grid=parameters_MLP_reg,
        n_jobs = -1,
        cv = 5
    )
    algo="MLP"

    mlp = grid_search_MLP_reg.fit(X_train, Y_train)
    Y_pred = mlp.predict(X_test)
    stop = timeit.default_timer()
    time_new = stop - start

    results = Add_reg_results(Y_pred, Y_test, Feat, algo, time_new, "Test")
    print("MLP finished running")
    return Y_pred, results, mlp


#******* Check for missing values in case of RVR *******#
class NanImputeScaler(BaseEstimator, TransformerMixin):
    """Scale an array with missing values, then impute them
    with a dummy value. This prevents the imputed value from impacting
    the mean/standard deviation computation during scaling.

    Parameters
    ----------
    with_mean : bool, optional (default=True)
        Whether to center the variables.

    with_std : bool, optional (default=True)
        Whether to divide by the standard deviation.

    nan_level : int or float, optional (default=-99.)
        The value to impute over NaN values after scaling the other features.
    """
    def __init__(self, with_mean=True, with_std=True, nan_level=-99.):
        self.with_mean = with_mean
        self.with_std = with_std
        self.nan_level = nan_level

    def fit(self, X, y=None):
        # Check the input array, but don't force everything to be finite.
        # This also ensures the array is 2D
        X = check_array(X, force_all_finite=False, ensure_2d=True)
        # compute the statistics on the data irrespective of NaN values
        self.means_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        return self

    def transform(self, X):
        # Check that we have already fit this transformer
        check_is_fitted(self, "means_")

        # get a copy of X so we can change it in place
        X = check_array(X, force_all_finite=False, ensure_2d=True)

        # center if needed
        if self.with_mean:
            X -= self.means_
        # scale if needed
        if self.with_std:
            X /= self.std_

        # now fill in the missing values
        X[np.isnan(X)] = self.nan_level
        return X


def Add_reg_results(Y_pred, Y_test, data, algo, time_new, set_type):
    dataset.append(data)
    Model.append(algo)
    MAE.append(round(mean_absolute_error(Y_test, Y_pred), 2))
    mse = round(mean_squared_error(Y_test, Y_pred), 2)
    MSE.append(mse)
    RMSE.append(round(math.sqrt(mse),2))
    R_2 = round(r2_score(Y_test, Y_pred),2)
    R_sq.append(R_2)
    run_time.append(time_new)
    Set_type.append(set_type)
    

    evaluation_metrics = pd.DataFrame({"Dataset": pd.Series(dataset), "Model": pd.Series(Model), "MAE": pd.Series(MAE), "RMSE": pd.Series(RMSE), "MSE": pd.Series(MSE),  
                                        "R_sq": pd.Series(R_sq), "Time":pd.Series(run_time), "Set":pd.Series(Set_type)})
    return evaluation_metrics


"""
This function performs the following function:
Input: takes the training set and splits required (should evenly divide the dataset)
-> performs k-fold cross validation, trains the ML algorithm, and computes the errors
Output: 
"""
def kFold_Cross_Val(X_train, Y_train, algo, sampling, data, k_splits):

    MAE_cv , MSE_cv , RMSE_cv , R_sq_cv = ([] for i in range(4))   

    # create an instance of the estimator or a pipeline containing the fit and predict methods
    pipeline = Build_Pipeline(algo)

    # create an object of the sampling strategy
    if (sampling=="Random"):
        ## KFold splits the training data into k consecutive subsets randomly
        kFold = KFold(n_splits=k_splits, shuffle=True)
        kfold = kFold.split(X_train, Y_train )
    elif (sampling=="Stratified"):
        ## Stratified KFold splits the training data into k subsets with uniform distribution 
        ## of the target class or group such as same no. of male and female in each subset
 
        kfold = get_k_age_range_stratified_partitions(k_splits,X_train, Y_train, True, 0.2, 10)
    
    for k, (train, test) in enumerate(kfold):
        # train and test represent the row indices of the train and validation subset 
        start = timeit.default_timer()

        pipeline.fit(X_train.iloc[train, :], Y_train.iloc[train])
        Y_pred = pipeline.predict(X_train.iloc[test, :])

        stop = timeit.default_timer()
        time_new = stop - start

        mae = mean_absolute_error(Y_pred, Y_train.iloc[test])
        mse = mean_squared_error(Y_pred, Y_train.iloc[test])
        rmse = round(math.sqrt(mse),2)
        r_2 = r2_score(Y_pred, Y_train.iloc[test])
        MAE_cv.append(mae)
        MSE_cv.append(mse)
        RMSE_cv.append(rmse)
        R_sq_cv.append(r_2)
    MAE.append(str(round(np.mean(MAE_cv),2)) +" ± " + str(round(np.std(MAE_cv),2)))
    MSE.append(str(round(np.mean(MSE_cv),2)) +" ± " + str(round(np.std(MSE_cv),2)))
    RMSE.append(str(round(np.mean(RMSE_cv),2)) +" ± " + str(round(np.std(RMSE_cv),2)))
    R_sq.append(str(round(np.mean(R_sq_cv),2)) +" ± " + str(round(np.std(R_sq_cv))))
    dataset.append(data)
    Model.append(algo)
    run_time.append(time_new)
    Set_type.append("Train")
    results = pd.DataFrame({"Dataset": pd.Series(dataset), "Model": pd.Series(Model), "MAE": pd.Series(MAE), "RMSE": pd.Series(RMSE), "MSE": pd.Series(MSE),  
                                        "R_sq": pd.Series(R_sq), "Time":pd.Series(run_time), "Set":pd.Series(Set_type)})
    return Y_pred, results
    

def Build_Pipeline(algo):
    if (algo=="LR"):
        pipeline = make_pipeline(StandardScaler(), LinearRegression())
    elif (algo=="LSVR"):
        pipeline = make_pipeline(StandardScaler(), LinearSVR(epsilon=5, C=20, fit_intercept=True))
    elif (algo=="GLM_Normal"):
        pipeline = make_pipeline(StandardScaler(), linear_model.TweedieRegressor())
    elif (algo=="GLM_Gamma"):
        pipeline = make_pipeline(StandardScaler(), linear_model.GammaRegressor(alpha=8))
    elif (algo=="RVR"):
        pipeline = Pipeline([("scale", NanImputeScaler()),("rvr", EMRVR(kernel="poly"))])
    return pipeline


def get_k_age_range_stratified_partitions(k, X, y, shuffle_data, test_size, num_bins=8):
    k_partitions = []
    """
    Divide ages into equal size bins and labels them
    """
    y = np.array(y)
    y_age_range = pd.qcut(y.reshape(y.shape[0]), num_bins, labels=False)

    if k == 1:
        shufflesplit = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=test_size)
        indx_split = list(shufflesplit.split(X, y_age_range))[0]
        k_partitions.append([indx_split[0], indx_split[1]])
    else:
        kfold = StratifiedKFold(n_splits=k, shuffle=shuffle_data, random_state=42)
        for tr_indx, tt_indx in kfold.split(X, y_age_range):
            shufflesplit = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=test_size)
            indx_split = list(shufflesplit.split(X[tt_indx], y_age_range[tt_indx]))[0]
            train_index = tt_indx[indx_split[0]]
            test_index = tt_indx[indx_split[1]]

            k_partitions.append([train_index, test_index])

    return k_partitions
    
  

def Tune_HyperParameters(X_train, X_test, Y_train, Y_test, algo):
    test_mae_list = []
    perc_within_eps_list = []

    eps = 5
    c_space = np.linspace(0.01, 10)

    for c in c_space:
        varied_svr = LinearSVR(epsilon=eps, C=c, fit_intercept=True, max_iter=10000)
        
        varied_svr.fit(X_train, Y_train)
        
        test_mae = mean_absolute_error(Y_test, varied_svr.predict(X_test))
        test_mae_list.append(test_mae)
        
        perc_within_eps = 100*np.sum(abs(Y_test-varied_svr.predict(X_test)) <= eps) / len(Y_test)
        perc_within_eps_list.append(perc_within_eps)

