# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:31:43 2020

@author: yuba
"""

import sys
sys.path.append(r'D:\work\back_test_system')
import DataBase as DB
import BackTest_2 as BT

import copy as c
import math as m
import numpy as np
import pandas as pd

import datetime

#%%

df = DB.getUnderlying(end_date='20190101')[['trade_date','close','pre_close','high','low']]
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

#%%

def rsi(symbol,window=14):
    df = c.deepcopy(symbol)
    df['T_1_close'] = df['close'].shift(1)
    df['max'] = df.apply(lambda x: max(x['close']-x['T_1_close'],0),axis=1)
    df['abs'] = df.apply(lambda x: abs(x['close']-x['T_1_close']),axis=1)
    alpha = 2/(1+window)
    df['RSI'] = df['max'].ewm(min_periods=window,adjust=False,alpha=alpha).mean()/\
        df['abs'].ewm(min_periods=window,adjust=False,alpha=alpha).mean()*100
    return df['RSI']

def adx(symbol,window=14):
    df = c.deepcopy(symbol)
    df['pDM'] = (df['high']-df['high'].shift(1)).apply(lambda x: max(x,0))
    df['nDM'] = (df['low'].shift(1)-df['low']).apply(lambda x: max(x,0))
    df['TR'] = df.apply(lambda x:max(x['high']-x['low'],abs(x['high']-x['pre_close']),abs(x['pre_close']-x['low'])),axis=1)
    df['ATR'] = df['TR'].rolling(window).mean()
    df['pDI'] = df['pDM'].rolling(window).mean()/df['ATR']*100
    df['nDI'] = df['nDM'].rolling(window).mean()/df['ATR']*100
    df['ADX'] = ((df['pDI']-df['nDI']).apply(abs)/(df['pDI']+df['nDI'])).rolling(window).mean()*100
    return df['ADX']

df['RSI'] = rsi(df).shift(1)
df['ADX'] = adx(df).shift(1)
df['pre_ADX'] = df['ADX'].shift(1)
df.dropna(axis=0,inplace=True)
df.reset_index(drop=True,inplace=True)

#%%

OpDB = pd.read_csv(r'D:\work\back_test_system\DataBase\Option\OP510050.csv')
OpDB[['trade_date','maturity_date','list_date']] = OpDB[['trade_date','maturity_date','list_date']].applymap(str)

def getOption(close,OpDB,td,maturity_date,level,CorP=True):
    CorP = 'C' if CorP else 'P'
    df = c.deepcopy(OpDB[(OpDB['trade_date']==td)&(OpDB['maturity_date']==maturity_date)&(OpDB['call_put']==CorP)])
    df['underlying_close'] = close
    df['dis'] = abs(df['exercise_price']-df['underlying_close'])
    df.reset_index(drop=True,inplace=True)
    if len(df)==0:
        return [np.nan,np.nan,np.nan,np.nan]
    index = df[df['dis']==df['dis'].min()].index[0]
    if CorP=='C':
        index = min(index+level,df.index[-1])
    else:
        index = max(index-level,df.index[0])
    return list(df.loc[index,['deposit','close']])

df['option'] = df.apply(lambda x: getOption(x['pre_close'],OpDB,x['trade_date'],x['maturity_date'],0,True),axis=1)
df[['C_deposit','C_op_close']] = df['option'].apply(pd.Series)
df['option'] = df.apply(lambda x: getOption(x['pre_close'],OpDB,x['trade_date'],x['maturity_date'],0,False),axis=1)
df[['P_deposit','P_op_close']] = df['option'].apply(pd.Series)

#%%

df['signal'] = df.apply(lambda x:1 if ((x['RSI']>=50 and x['RSI']<80) or (x['RSI']<20)) and (x['ADX']>=x['pre_ADX']) else \
                        (-1 if (x['RSI']>=80) and (x['ADX']<x['pre_ADX']) else 0),axis=1)
#df['signal'] = df.apply(lambda x:1 if ((x['RSI']>=50 and x['RSI']<80) or (x['RSI']<20)) else \
#                        (-1 if (x['RSI']>=80) else 0),axis=1)
Signal,Price,Direction,Deposit,flag,cover = [],[],[],[],False,True
Volume,Position,Pct = [],[],[]
n = len(df)
for i in range(n):
    if flag:
        if (i==n-1) or (df['maturity_date'].loc[i]!=df['maturity_date'].loc[i+1]): # 到期前平仓
            signal = -1
            price = [df['C_op_close'].loc[i] if cover else df['P_op_close'].loc[i]]
            direction = [-1]
            deposit = [df['C_deposit'].loc[i] if cover else df['P_deposit'].loc[i]]
            flag = False
        else:
            if cover:
                if df['signal'].loc[i]==1:
                    signal = -1
                    price = [df['C_op_close'].loc[i]]
                    direction = [-1]
                    deposit = [df['C_deposit'].loc[i]]
                    flag = False
                else:
                    signal = 0
                    price = [df['C_op_close'].loc[i]]
                    direction = [0]
                    deposit = [df['C_deposit'].loc[i]]
            else:
                if df['signal'].loc[i]==-1:
                    signal = -1
                    price = [df['P_op_close'].loc[i]]
                    direction = [-1]
                    deposit = [df['P_deposit'].loc[i]]
                    flag = False
                else:
                    signal = 0
                    price = [df['P_op_close'].loc[i]]
                    direction = [0]
                    deposit = [df['P_deposit'].loc[i]]
    else:
        if df['signal'].loc[i]==-1:
            signal = 1
            price = [df['C_op_close'].loc[i]]
            direction = [-1]
            deposit = [df['C_deposit'].loc[i]]
            flag,cover = True,True
        elif df['signal'].loc[i]==1:
            signal = 1
            price = [df['P_op_close'].loc[i]]
            direction = [-1]
            deposit = [df['P_deposit'].loc[i]]
            flag,cover = True,False
        else:
            signal = 0
            price = [0]
            direction = [0]
            deposit = [0]
    Signal.append(signal)
    Price.append(price)
    Direction.append(direction)
    Deposit.append(deposit)
    Volume.append([-1])
    Position.append([0])
    Pct.append([0])

signalDf = pd.DataFrame({'trade_date':df['trade_date'],'signal':Signal,'price':Price,'direction':Direction,'volume':Volume,'deposit':Deposit,'position':Position,'pct':Pct})
signalDf['trade_date'] = signalDf['trade_date'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m%d'))

#%%

recordDf = BT.OptionBT(signalDf)
benchmarkDf = c.deepcopy(df[['trade_date','close']])
benchmarkDf['trade_date'] = benchmarkDf['trade_date'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m%d'))
stat = BT.Visualize(recordDf,benchmarkDf)