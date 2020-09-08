import numpy as np
import pandas as pd
def prior(correlation,covarince):
    '''
    Computes prior matrix for shrinkage approach.
    Prior is constant correlation matrix with variance in diagonal
    :param correlation: correlation matrix
    :param covariance: covariance matrix
    :return: prior
    '''
    constant = correlation.mean().mean()
    # square matrix with mean correlation
    prior_matrix = np.array(np.multiply(np.divide(covarince,correlation),constant))
    # fill diagonal with variance
    diagonal = np.diagonal(covarince)
    for i,elem in enumerate(diagonal):
        prior_matrix[i,i]=elem
    return prior_matrix