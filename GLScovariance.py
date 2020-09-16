import numpy as np
import pandas as pd
from Prior import prior
from Shrinkage import shrinkage


def GLScovariance(returns, factor, window):
    '''
    Generalised least squares (weighted)
    Cross sectional linear regression (GLS) between factor(t-1) and returns(t)
    Factor matrix must be adjusted/shifted to match return row

    :param returns: Asset returns (TxN)
    :param factor: Factor matrix (TxN)
    :param window: trailling window for covariance matrix (TxN)
    :return: gamma, tstat
    '''
    T = returns.shape[0]
    # Initiate gamma vector
    gamma = np.array([np.nan] * (T-window))
    # index for gamma vector
    m=0
    # cross sectional regression
    for i in range(window,T):
        # Recalculate covariance,prior and shrinkage every new month (not daily to save computing time)
        # find column names with NaN in return and factor matrices
        col1 = list(returns.iloc[i, :][returns.iloc[i, :].isnull()].index)
        col2 = list(factor.iloc[i, :][factor.iloc[i, :].isnull()].index)
        if i == window or returns.index[i].month != returns.index[i-1].month or (set(col1+col2)!=col1_2_previous and i!=window):
            # find columns to delete (columns with at least one NaN value
            col3 = list(returns.iloc[i-window:i,].columns[returns.iloc[i-window:i,].isnull().any()])
            returns_trimed = returns.iloc[i-window:i,].drop(labels=set(col1+col2+col3),axis=1)
            # Covariance matrix
            cov_matrix = returns_trimed.cov()
            # Prior matrix
            prior_matrix = prior(returns_trimed.corr(),cov_matrix)
            cov_shrink = shrinkage(np.array(returns_trimed),np.array(cov_matrix),prior_matrix)

        # join column names
        delete_cols = set(col1 + col2 + col3)
        # remove columns with NaN from return and factor matrices
        returns_reg = returns.iloc[i, :].drop(index=delete_cols)
        # add intercept to factor matrix
        factor_reg = np.array([[1] * len(returns_reg)] + [list(factor.iloc[i, :].drop(index=delete_cols))])

        # perform regression
        temp = np.linalg.inv(factor_reg @ np.linalg.inv(cov_shrink) @ np.transpose(factor_reg)) @ factor_reg @ np.linalg.inv(cov_shrink) @ np.transpose(returns_reg)

        gamma[m] = temp[1]
        m+=1
        col1_2_previous=set(col1+col2)

    tstat = np.nanmean(gamma) / (np.nanstd(gamma) / np.sqrt(len(gamma)))
    return gamma,tstat