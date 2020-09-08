import numpy as np
import pandas as pd
import datetime
from scipy.stats import linregress
from OLSfamaMacBeth import OLSfama
from GLSweightFamaMacBeth import GLSweightFama
from GLScovariance import GLScovariance
import matplotlib.pyplot as plt

#----------------------------------------------IMPORT CSV FILES---------------------------------------------------------
dates = pd.read_csv('UK_Dates.csv',header=None,parse_dates=[0],index_col=0)
names = pd.read_csv('UK_Names.csv',header=None)
returns = pd.read_csv('UK_Returns.csv',header=None,na_values='NaN')
returns.columns=names[0].tolist()
returns.set_index(dates.index,inplace=True,drop=True)
live_names = pd.read_csv('UK_live.csv',header=None,na_values='NaN')
live_names.columns=names[0].tolist()
live_names.set_index(dates.index,inplace=True,drop=True)
marketCap = pd.read_csv('UK_MarketValues.csv',header=None,na_values='NaN')
marketCap.columns=names[0].tolist()
marketCap.set_index(dates.index,inplace=True,drop=True)




#----------------------------------------------FACTOR MATRIX------------------------------------------------------------
# Factor matrix has highest daily return over month. So number of observations is number of months
# Not use last row because its the beginning of different month
factor = returns[:-1].groupby([returns[:-1].index.year,returns[:-1].index.month]).max()
average_factor = factor.mean(axis=1,skipna=True)
std_factor = factor.std(axis=1,skipna=True)
factor_norm_monthly = (factor.subtract(average_factor,axis='index')).divide(std_factor,axis='index')

# Shift factor return by one month. Factor in Jan shifted to February
factor_norm_daily = pd.DataFrame(np.nan,index=returns.index,columns=returns.columns)
start=0
for date_idx,row in factor_norm_daily.iterrows():
    # skip rows 1st month available because there is no factor values
    if date_idx.year == factor_norm_monthly.index[0][0] and date_idx.month == factor_norm_monthly.index[0][1]:
        start +=1
    else:
        # Get date to use in monthly factor matrix
        if date_idx.month != 1:
            temp = datetime.date(date_idx.year,date_idx.month-1,1)
        else:
            temp = datetime.date(date_idx.year-1, 12, 1)
        # update daily factor matrix row with factor from previous month
        row[:] = factor_norm_monthly.loc[(temp.year,temp.month),]


#----------------------------------------------REGRESSIONS--------------------------------------------------------------
# OLS regression
gamma_OLS,tstat_OLS=OLSfama(returns.iloc[start:,],factor_norm_daily.iloc[start:,])
print(gamma_OLS)
print(tstat_OLS)



# GLS Market Cap weighted regression

weight_MarketCap = marketCap.divide(marketCap.sum(axis=1),axis='index')
gamma_GLSweight,tstat_GLSweight = GLSweightFama(returns.iloc[start:,],
                                                factor_norm_daily.iloc[start:,],
                                                weight_MarketCap.iloc[start-1:-1,])
print(tstat_GLSweight)

# GLS covariance matrix (shrinkage approach)
window = 500
gamma_GLScovariance, tstat_GLScovariance = GLScovariance(returns.iloc[start:,],factor_norm_daily.iloc[start:,],window)
print(tstat_GLScovariance)

#-----------------------------------------------------Plot--------------------------------------------------------------

plt.plot(returns.index[start:],np.cumsum(gamma_OLS),color='r',label='OLS')
plt.plot(returns.index[start:],np.cumsum(gamma_GLSweight),color='b',label='GLS weight')
plt.plot(returns.index[start+window:],np.cumsum(gamma_GLScovariance),color='g',label='GLS covariance')
plt.xlabel('Date')
plt.ylabel('Cumulative factor return')
plt.legend()
plt.show()