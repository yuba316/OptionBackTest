# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:23:43 2020

@author: yuba316
"""

import copy
import sys
sys.path.append(r'D:\work\back_test_system')
import DataBase as DB
from scipy.stats import norm
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

#%%

def BSMDelta(S,K,sigma,t,rf,CorP=True):
    d1 = (np.log(S/K)+(rf+0.5*sigma*sigma)*t)/(sigma*np.sqrt(t))
    return norm.cdf(d1)-(not CorP)

def BSMGamma(S,K,sigma,t,rf,CorP=True):
    temp = sigma*np.sqrt(t)
    d1 = (np.log(S/K)+(rf+0.5*sigma*sigma)*t)/temp
    return norm.pdf(d1)/(S*temp)

def BSMVega(S,K,sigma,t,rf,CorP=True):
    temp = np.sqrt(t)
    d1 = (np.log(S/K)+(rf+0.5*sigma*sigma)*t)/(sigma*temp)
    return S*norm.pdf(d1)*temp

def BSMTheta(S,K,sigma,t,rf,CorP=True):
    a = np.sqrt(t)
    b = sigma*a
    d1 = (np.log(S/K)+(rf+0.5*sigma*sigma)*t)/b
    d2 = d1-b
    return -0.5*S*norm.pdf(d1)*sigma/a+\
        (-1)**CorP*rf*K*np.exp(-rf*t)*norm.cdf((-1)**(not CorP)*d2)

def BSMRho(S,K,sigma,t,rf,CorP=True):
    sign = (-1)**(not CorP)
    temp = sigma*np.sqrt(t)
    d1 = (np.log(S/K)+(rf+0.5*sigma*sigma)*t)/temp
    d2 = sign*(d1-temp)
    return sign*K*t*np.exp(-rf*t)*norm.cdf(d2)

def BSMGreeks(S,K,sigma,t,rf,CorP=True):
    sign = (-1)**(not CorP)
    a = np.sqrt(t)
    b = sigma*a
    d1 = (np.log(S/K)+(rf+0.5*sigma*sigma)*t)/b
    d2 = d1-b
    N1,N2,Nd1 = norm.cdf(d1),norm.cdf(d2),norm.pdf(d1)
    Greeks = {}
    Greeks['Delta'] = N1-(not CorP)
    Greeks['Gamma'] = Nd1/(S*b)
    Greeks['Vega'] = S*Nd1*a
    Greeks['Theta'] = -0.5*S*Nd1*sigma/a+(-1)**CorP*rf*K*np.exp(-rf*t)*N2
    Greeks['Rho'] = sign*K*t*np.exp(-rf*t)*norm.cdf(sign*d2)
    return pd.Series(Greeks)

def BSMPricing(S,K,sigma,t,rf,CorP=True):
    sign = (-1)**(not CorP)
    temp = sigma*np.sqrt(t)
    d1 = sign*((np.log(S/K)+(rf+0.5*sigma*sigma)*t)/temp)
    d2 = sign*(d1-temp)
    return sign*(S*norm.cdf(d1)-K*np.exp(-rf*t)*norm.cdf(d2))

def BSMVanna(S,K,sigma,t,rf,CorP=True): # delta关于隐含波动率的导数
    temp = sigma*np.sqrt(t)
    d1 = (np.log(S/K)+(rf+0.5*sigma*sigma)*t)/temp
    d2 = d1-temp
    return -norm.pdf(d1)*norm.cdf(d2)/sigma

def BSMCharm(S,K,sigma,t,rf,CorP=True): # delta关于时间的导数
    temp = sigma*np.sqrt(t)
    d1 = (np.log(S/K)+(rf+0.5*sigma*sigma)*t)/temp
    d2 = d1-temp
    return -norm.pdf(d1)*(rf/temp-0.5*d2/t)

def BSMVomma(S,K,sigma,t,rf,CorP=True): # vega关于隐含波动率的导数
    a = np.sqrt(t)
    b = sigma*a
    d1 = (np.log(S/K)+(rf+0.5*sigma*sigma)*t)/b
    d2 = d1-b
    return S*norm.pdf(d1)*a*d1*d2/sigma

def BSMVeta(S,K,sigma,t,rf,CorP=True): # vega关于时间的导数
    a = np.sqrt(t)
    b = sigma*a
    d1 = (np.log(S/K)+(rf+0.5*sigma*sigma)*t)/b
    d2 = d1-b
    return S*norm.pdf(d1)*a*(rf*d1/b-0.5*(1+d1*d2)/t)

def BSMSpeed(S,K,sigma,t,rf,CorP=True): # gamma对标的物价格的导数
    temp = sigma*np.sqrt(t)
    d1 = (np.log(S/K)+(rf+0.5*sigma*sigma)*t)/temp
    return -norm.pdf(d1)/(np.square(S)*temp)*(d1/temp+1)

def BSMZomma(S,K,sigma,t,rf,CorP=True): # gamma对隐含波动率的导数
    temp = sigma*np.sqrt(t)
    d1 = (np.log(S/K)+(rf+0.5*sigma*sigma)*t)/temp
    d2 = d1-temp
    return norm.pdf(d1)*(d1*d2-1)/(S*temp*sigma)

def BSMColor(S,K,sigma,t,rf,CorP=True): # gamma对时间的导数
    temp = sigma*np.sqrt(t)
    d1 = (np.log(S/K)+(rf+0.5*sigma*sigma)*t)/temp
    d2 = d1-temp
    return 0.5*norm.pdf(d1)*(1+2*rf*t*d1/temp-d1*d2)/(S*t*temp)

#%% test

df = pd.DataFrame({'Sigma':np.linspace(0.01,0.29,92),'T':np.arange(1,93,1)})
df['ITM'],df['ATM'],df['OTM'],df['K'],df['sigma'],df['t'],df['rf'],df['CorP'] = 2.32,2.3,2.29,2.3,0.05,30/365,0.035,True

def Visualize(stock,x,y):
    
    df = copy.deepcopy(stock)
    if y=='delta':
        if x=='t':
            df['delta_t_i'] = df.apply(lambda x: BSMDelta(x['ITM'],x['K'],x['sigma'],x['T']/365,x['rf']),axis=1)
            df['delta_t_a'] = df.apply(lambda x: BSMDelta(x['ATM'],x['K'],x['sigma'],x['T']/365,x['rf']),axis=1)
            df['delta_t_o'] = df.apply(lambda x: BSMDelta(x['OTM'],x['K'],x['sigma'],x['T']/365,x['rf']),axis=1)
        elif x=='sigma':
            df['delta_sigma_i'] = df.apply(lambda x: BSMDelta(x['ITM'],x['K'],x['Sigma'],x['t'],x['rf']),axis=1)
            df['delta_sigma_a'] = df.apply(lambda x: BSMDelta(x['ATM'],x['K'],x['Sigma'],x['t'],x['rf']),axis=1)
            df['delta_sigma_o'] = df.apply(lambda x: BSMDelta(x['OTM'],x['K'],x['Sigma'],x['t'],x['rf']),axis=1)
    elif y=='gamma':
        if x=='t':
            df['gamma_t_i'] = df.apply(lambda x: BSMGamma(x['ITM'],x['K'],x['sigma'],x['T']/365,x['rf']),axis=1)
            df['gamma_t_a'] = df.apply(lambda x: BSMGamma(x['ATM'],x['K'],x['sigma'],x['T']/365,x['rf']),axis=1)
            df['gamma_t_o'] = df.apply(lambda x: BSMGamma(x['OTM'],x['K'],x['sigma'],x['T']/365,x['rf']),axis=1)
        elif x=='sigma':
            df['gamma_sigma_i'] = df.apply(lambda x: BSMGamma(x['ITM'],x['K'],x['Sigma'],x['t'],x['rf']),axis=1)
            df['gamma_sigma_a'] = df.apply(lambda x: BSMGamma(x['ATM'],x['K'],x['Sigma'],x['t'],x['rf']),axis=1)
            df['gamma_sigma_o'] = df.apply(lambda x: BSMGamma(x['OTM'],x['K'],x['Sigma'],x['t'],x['rf']),axis=1)
    elif y=='vega':
        if x=='t':
            df['vega_t_i'] = df.apply(lambda x: BSMVega(x['ITM'],x['K'],x['sigma'],x['T']/365,x['rf']),axis=1)
            df['vega_t_a'] = df.apply(lambda x: BSMVega(x['ATM'],x['K'],x['sigma'],x['T']/365,x['rf']),axis=1)
            df['vega_t_o'] = df.apply(lambda x: BSMVega(x['OTM'],x['K'],x['sigma'],x['T']/365,x['rf']),axis=1)
        elif x=='sigma':
            df['vega_sigma_i'] = df.apply(lambda x: BSMVega(x['ITM'],x['K'],x['Sigma'],x['t'],x['rf']),axis=1)
            df['vega_sigma_a'] = df.apply(lambda x: BSMVega(x['ATM'],x['K'],x['Sigma'],x['t'],x['rf']),axis=1)
            df['vega_sigma_o'] = df.apply(lambda x: BSMVega(x['OTM'],x['K'],x['Sigma'],x['t'],x['rf']),axis=1)
    
    if x=='t':
        plt.title(y+'-到期日')
        plt.plot(-1*df['T'],df[y+'_t_i'],label='价内')
        plt.plot(-1*df['T'],df[y+'_t_a'],label='平价')
        plt.plot(-1*df['T'],df[y+'_t_o'],label='价外')
        plt.legend(loc='upper right')
    elif x=='sigma':
        plt.title(y+'-波动率')
        plt.plot(df['Sigma'],df[y+'_sigma_i'],label='价内')
        plt.plot(df['Sigma'],df[y+'_sigma_a'],label='平价')
        plt.plot(df['Sigma'],df[y+'_sigma_o'],label='价外')
        plt.legend(loc='upper right')

plt.figure(figsize=(18,8))
plt.subplot(231)
Visualize(df,'t','delta')
plt.subplot(234)
Visualize(df,'sigma','delta')
plt.subplot(232)
Visualize(df,'t','gamma')
plt.subplot(235)
Visualize(df,'sigma','gamma')
plt.subplot(233)
Visualize(df,'t','vega')
plt.subplot(236)
Visualize(df,'sigma','vega')

#%%

'''
df = DB.getUnderlying()[['trade_date','close']]
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
rfDf = DB.getRf(df['trade_date'].iloc[0],df['trade_date'].iloc[-1])
df = pd.merge(df,rfDf,how='left',on='trade_date')
start_date = df['trade_date'].iloc[0]
end_date = df['trade_date'].iloc[-2]
option_basic = DB.getOpBasic()
maturity_date = list(option_basic[(option_basic['maturity_date']>start_date)&(option_basic['maturity_date']<end_date)]['maturity_date'].unique())
maturity_date.append(list(option_basic[option_basic['maturity_date']>maturity_date[-1]]['maturity_date'].unique())[0])
df['maturity_date'] = np.nan
df.iloc[df[df['trade_date'].apply(lambda x: x in maturity_date)].index,-1] = maturity_date[:-1]
df['maturity_date'] = df['maturity_date'].shift(-5) # 提前5个交易日平仓
df.iloc[-1,-1] = maturity_date[-1]
df.fillna(method='bfill',inplace=True)

OpDB = pd.read_csv(r'D:\work\back_test_system\DataBase\Option\OP510050.csv')
OpDB[['trade_date','maturity_date','list_date']] = OpDB[['trade_date','maturity_date','list_date']].applymap(str)

def getKT(close,OpDB,td,maturity_date,rf,CorP=True):
    CorP = 'C' if CorP else 'P'
    df = copy.deepcopy(OpDB[(OpDB['trade_date']==td)&(OpDB['maturity_date']==maturity_date)])
    df['underlying_close'] = close
    T = ((datetime.datetime.strptime(maturity_date,'%Y%m%d')-datetime.datetime.strptime(td,'%Y%m%d')).days+1)/365
    df['dis'] = abs(df['exercise_price']-df['underlying_close'])
    K = df[(df['dis']==df['dis'].min())&(df['call_put']==CorP)]['exercise_price']
    if len(K)==0:
        return [np.nan,T]
    return [K.iloc[0],T]

df['KT'] = df.apply(lambda x: getKT(x['close'],OpDB,x['trade_date'],x['maturity_date'],x['rf']),axis=1)
df[['K','t']] = df['KT'].apply(pd.Series)
df['sigma'] = df['close'].rolling(21).std()
df.dropna(axis=0,inplace=True)
df[['Delta','Gamma','Vega','Theta','Rho']] = df.apply(lambda x: BSMGreeks(x['close'],x['K'],x['sigma'],x['t'],x['rf']),axis=1)
'''