import numpy as np
import pandas as pd


def GLSweightFama(returns, factor, weight):
    '''
    Generalised least squares (weighted)
    Cross sectional linear regression (GLS) between factor(t-1) and returns(t)
    Factor matrix must be adjusted/shifted to match return row

    :param returns: Asset returns (TxN)
    :param factor: Factor matrix (TxN)
    :param weight: weight matrix (TxN)
    :return: gamma, tstat
    '''
    T = returns.shape[0]
    # Initiate gamma vector
    gamma = np.array([np.nan] * T)
    # cross sectional regression
    for i in range(T):
        # find column names with NaN in return and factor matrices
        col1 = list(returns.iloc[i, :][returns.iloc[i, :].isnull()].index)
        col2 = list(factor.iloc[i, :][factor.iloc[i, :].isnull()].index)
        col3 = list(weight.iloc[i, :][weight.iloc[i, :].isnull()].index)    #market cap missing sometimes so delete cols
        # join column names
        delete_cols = set(col1 + col2 + col3)
        # remove columns with NaN from return and factor matrices
        returns_reg = returns.iloc[i, :].drop(index=delete_cols)
        # add intercept to factor matrix
        factor_reg = np.array([[1]*len(returns_reg)]+[list(factor.iloc[i, :].drop(index=delete_cols))])
        # Weight diagonal matrix
        diag_weight = np.diag(weight.iloc[i,:].drop(index=delete_cols))
        # perform regression
        temp = np.linalg.inv(factor_reg @ diag_weight @ factor_reg.transpose()) @ factor_reg @ diag_weight @ np.array(returns_reg).transpose()
        gamma[i]=temp[1]
    tstat = np.nanmean(gamma) / (np.nanstd(gamma) / np.sqrt(len(gamma)))
    return gamma, tstat