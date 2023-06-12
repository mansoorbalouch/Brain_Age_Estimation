
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
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.svm import LinearSVR
from sklearn_rvm import EMRVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import timeit
import warnings

warnings.filterwarnings('ignore')



dataset = []
Model = []
MAE = []
MSE = []
RMSE = []
R_sq = []
run_time = []

print("Package import done")

####################### Linear Regression ####################################
def LR(X_train, Y_train, X_test, Y_test, Feat):
    print("LR started running")
    start = timeit.default_timer()
    lm = LinearRegression()

    lm.fit(X_train, Y_train)

    # print("Intercept: {:,.3f}".format(lm.intercept_))
    # print("Coefficient: {:,.3f}".format(lm.coef_[1]))

    dataset.append(Feat)
    Model.append("Linear Regression")
    Y_pred = lm.predict(X_test)
    stop = timeit.default_timer()
    time_new = stop - start

    MAE.append(round(mean_absolute_error(Y_test, Y_pred), 2))

    mse = round(mean_squared_error(Y_test, Y_pred), 2)
    MSE.append(mse)
    RMSE.append(round(math.sqrt(mse),2))
    R_2 = round(sklearn.metrics.r2_score(Y_test, Y_pred),2)
    R_sq.append(R_2)
    run_time.append(time_new)
    results= Add_reg_results()
    print("LR finished running")
    return Y_pred, results


################## Linear SVR  ###########################################
def LSVR(X_train, Y_train, X_test, Y_test, Feat):
    print("SVR started running")
    start = timeit.default_timer()
    eps = 5
    svr = LinearSVR(epsilon=eps, C=0.14, fit_intercept=True)
    svr.fit(X_train, Y_train)

    # storing the results
    dataset.append(Feat)
    Model.append("Linear Support Vector Regression")
    Y_pred = svr.predict(X_test)
    stop = timeit.default_timer()
    time_new = stop - start

    MAE.append(round(mean_absolute_error(Y_test, Y_pred), 2))

    mse = round(mean_squared_error(Y_test, Y_pred), 2)
    MSE.append(mse)
    RMSE.append(round(math.sqrt(mse),2))
    R_2 = round(sklearn.metrics.r2_score(Y_test, Y_pred),2)
    R_sq.append(R_2)
    run_time.append(time_new)
    results = Add_reg_results()
    print("SVR finished running")
    return Y_pred, results


################## Relevance Vector Regression  ############################
def RVR(X_train, Y_train, X_test, Y_test, Feat):
    print("RVR started running")
    start = timeit.default_timer()
    pipe = Pipeline([
        ("scale", NanImputeScaler()),
        ("rvr", EMRVR(kernel="poly"))
    ]).fit(X_train, Y_train)

    dataset.append(Feat)
    Model.append("Relevance Vector Regression")
    Y_pred = pipe.predict(X_test)
    stop = timeit.default_timer()
    time_new = stop - start

    MAE.append(round(mean_absolute_error(Y_test, Y_pred), 2))

    mse = round(mean_squared_error(Y_test, Y_pred), 2)
    MSE.append(mse)
    RMSE.append(round(math.sqrt(mse),2))
    R_2 = round(sklearn.metrics.r2_score(Y_test, Y_pred),2)
    R_sq.append(R_2)
    run_time.append(time_new)
    results = Add_reg_results()
    print("RVR finished running")
    return Y_pred, results


################ Generalized Linear Model ################################

#********* Gamma Distribution (positive skewed) ********#

def GLM_Gamma(X_train, Y_train, X_test, Y_test, Feat):
    print("GLM_Gamma started running")
    start = timeit.default_timer()
    glm_gamma = linear_model.GammaRegressor(alpha=8)
    glm_gamma.fit(X_train, Y_train)
    dataset.append(Feat)
    Model.append("Generalized Linear Model (Gamma)")
    Y_pred = glm_gamma.predict(X_test)
    stop = timeit.default_timer()
    time_new = stop - start
    mae = round(mean_absolute_error(Y_test, Y_pred), 2)
    MAE.append(round(mean_absolute_error(Y_test, Y_pred), 2))

    mse = round(mean_squared_error(Y_test, Y_pred), 2)
    MSE.append(mse)
    RMSE.append(round(math.sqrt(mse),2))
    R_2 = round(sklearn.metrics.r2_score(Y_test, Y_pred),2)
    R_sq.append(R_2)
    run_time.append(time_new)
    # results = Add_reg_results()
    print("GLM_Gamma finished running")
    return Y_pred


#********* Tweedie Regressor (Normal Distribution) ********#

def GLM_normal(X_train, Y_train, X_test, Y_test, Feat):
    print("GLM_Normal started running")
    start = timeit.default_timer()

    glm = linear_model.TweedieRegressor()
    glm.fit(X_train, Y_train)
    dataset.append(Feat)
    Model.append("Generalized Linear Model (Normal)")
    Y_pred = glm.predict(X_test)
    stop = timeit.default_timer()
    time_new = stop - start

    MAE.append(round(mean_absolute_error(Y_test, Y_pred), 2))

    mse = round(mean_squared_error(Y_test, Y_pred), 2)
    MSE.append(mse)
    RMSE.append(round(math.sqrt(mse),2))
    R_2 = round(sklearn.metrics.r2_score(Y_test, Y_pred),2)
    R_sq.append(R_2)
    run_time.append(time_new)
    results = Add_reg_results()
    print("GLM_Normal finished running")
    return Y_pred, results



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


def Add_reg_results():
    dataset_s = pd.Series(dataset)
    Model_s = pd.Series(Model)
    MAE_s = pd.Series(MAE)
    RMSE_s = pd.Series(RMSE)
    R_sq_s = pd.Series(R_sq)
    MSE_s = pd.Series(MSE)
    time_s = pd.Series(run_time)
    evaluation_metrics = pd.DataFrame({"Dataset": dataset_s, "Model": Model_s, "RMSE": RMSE_s, "MSE": MSE_s, "MAE": MAE_s,  "R_sq": R_sq_s, "Time":time_s})
    return evaluation_metrics


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


