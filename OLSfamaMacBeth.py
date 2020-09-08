from scipy.stats import linregress
import numpy as np
def OLSfama(returns,factor):
    '''
    Ordinary least squares regression
    Cross sectional linear regression (OLS) between factor(t-1) and returns(t)
    Factor matrix must be adjusted/shifted to match return row

    :param returns: Asset returns (TxN)
    :param factor: Factor matrix (TxN)
    :return: gamma, tstat
    '''
    T = returns.shape[0]
    # Initiate gamma vector
    gamma = np.array([np.nan]*T)
    # cross sectional regression
    for i in range(T):
        # find column names with NaN in return and factor matrices
        col1 = list(returns.iloc[i,:][returns.iloc[i,:].isnull()].index)
        col2 = list(factor.iloc[i,:][factor.iloc[i, :].isnull()].index)
        # join column names
        delete_cols = set(col1+col2)
        # remove columns with NaN from return and factor matrices
        returns_reg = returns.iloc[i,:].drop(index=delete_cols)
        factor_reg = factor.iloc[i,:].drop(index=delete_cols)
        # perform regression
        gamma[i] = linregress(factor_reg,returns_reg)[0]
    tstat = np.nanmean(gamma)/(np.nanstd(gamma)/np.sqrt(T))
    return gamma,tstat