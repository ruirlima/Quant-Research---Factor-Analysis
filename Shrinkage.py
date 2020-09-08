import numpy as np
import pandas as pd

def shrinkage(returns,covariance,prior):
    '''
    Ledoit and Wold shrinkage approach

    :param returns: asset returns
    :param covariance: covariance matrix
    :param prior: prior
    :return: shrinkage matrix
    '''
    T = returns.shape[0]         #number of observations
    N = returns.shape[1]         #number of assets
    d = 1/N*np.linalg.norm(np.subtract(covariance,prior),'fro')**2
    y = np.power(returns,2)
    r2 = 1/N/T**2*np.sum(np.sum(np.matmul(y.transpose(),y)))-1/N/T*np.sum(np.sum(np.power(covariance,2)))
    shrink = max(0,min(1,r2/d))
    cov_shrink = np.multiply(shrink,prior)+np.multiply(1-shrink,covariance)
    return cov_shrink
