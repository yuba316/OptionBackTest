# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:21:00 2020

@author: yuba316
"""

import sys
sys.path.append(r'D:\work\back_test_system')
import DataBase as DB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% BSM标准股票轨道

def getT(sample,step):
    zero = np.zeros(sample)
    T = np.ones((sample,step-1))
    T = T.cumsum(1)
    T = np.insert(T,0,values=zero,axis=1)
    return T/252

def getBW(sample,step,miu=0,sigma=1): # 得到sample个step步的布朗运动过程，起点0算一步
    step = step-1
    zero = np.zeros(sample)
    W = np.random.normal(miu,sigma,sample*step)
    W = np.reshape(W,(sample,step))
    W = W.cumsum(1)
    W = np.insert(W,0,values=zero,axis=1)
    return W

def getSP_BSM(sample,step,S0,miu,sigma): # BSM标准股票轨道
    T = getT(sample,step)
    W = getBW(sample,step)
    St = S0*np.exp((miu-0.5*np.square(sigma))*T+sigma*W)
    return St

def getProfit(St): # 取得模拟股票轨道的对数收益率
    (row,column) = St.shape
    r = np.log(St[:,1:]/St[:,:-1])
    r = np.reshape(r,row*(column-1))
    return r

#%% MDJ带泊松跳过程的股票轨道

def getPS(sample,step,lam=1): # 累计泊松分布随机数
    step,lam = step-1,lam/252
    zero = np.zeros(sample)
    P = np.random.poisson(lam,sample*step)
    P = np.reshape(P,(sample,step))
    P = P.cumsum(1)
    P = np.insert(P,0,values=zero,axis=1)
    return P

def getSP_MDJ(sample,step,S0,miu,sigma,lam,Jmiu,Jsigma):
    k = np.exp(Jmiu+0.5*np.square(Jsigma))-1
    T = getT(sample,step)
    W = getBW(sample,step)
    P = getPS(sample,step,lam)
    JW = getBW(sample,P.max()+1,Jmiu,Jsigma)
    Y = np.zeros((sample,step))
    for i in range(sample):
        Y[i,:] = JW[i,P[i,:]]
    St = S0*np.exp((miu-0.5*np.square(sigma)-lam*k)*T+sigma*W+Y)
    return St

#%% VG方差-Gamma跳过程的股票轨道

def getGamma(sample,step,miu,sigma):
    step = step-1
    alpha = (np.square(miu)/sigma)/252
    beta = miu/sigma
    zero = np.zeros(sample)
    G = np.random.gamma(alpha,beta,sample*step)
    G = np.reshape(G,(sample,step))
    G = G.cumsum(1)
    G = np.insert(G,0,values=zero,axis=1)
    G = np.round(G)
    G = G.astype(np.int16)
    return G

def getSP_VG(sample,step,S0,miu,sigma,Gmiu,Gsigma):
    w = 1/Gsigma*np.log(1-0.5*np.square(sigma)*Gsigma)
    T = getT(sample,step)
    G = getGamma(sample,step,Gmiu,Gsigma)
    W = getBW(sample,G.max()+1)
    GW = np.zeros((sample,step))
    for i in range(sample):
        GW[i,:] = W[i,G[i,:]]
    St = S0*np.exp((miu+w)*T+sigma*GW)
    return St

#%% test

df = DB.getUnderlying()[['trade_date','close','pre_close']]
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df['profit'] = np.log(df['close']/df['pre_close'])

miu = df['profit'].iloc[-252:].mean()
sigma = df['profit'].iloc[-252:].std()
S0,sample,step = df['close'].iloc[-1],100,252

#%% BSM标准股票轨道

St = getSP_BSM(sample,step,S0,miu,sigma)
n = range(step)
plt.figure()
for i in range(sample):
    plt.plot(n,St[i,:])
r = getProfit(St)
plt.figure()
sns.set(style='white')
sns.distplot(r,color='orange',kde=True,hist=True,kde_kws={'shade':True,'color': 'darkorange','facecolor':'gray'})

#%% MDJ带泊松跳过程的股票轨道

lam,Jmiu,Jsigma = 1,-0.1,0.1
St = getSP_MDJ(sample,step,S0,miu,sigma,lam,Jmiu,Jsigma)
n = range(step)
plt.figure()
for i in range(sample):
    plt.plot(n,St[i,:])
r = getProfit(St)
plt.figure()
sns.set(style='white')
sns.distplot(r,color='orange',kde=True,hist=True,kde_kws={'shade':True,'color': 'darkorange','facecolor':'gray'})

#%% VG方差-Gamma跳过程的股票轨道

Gmiu,Gsigma = 1,0.1
St = getSP_VG(sample,step,S0,miu,sigma,Gmiu,Gsigma)
n = range(step)
plt.figure()
for i in range(sample):
    plt.plot(n,St[i,:])