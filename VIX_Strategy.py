# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 08:48:46 2020

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

df = DB.getUnderlying(end_date='20190101')[['trade_date','close','pre_close']]
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
df['maturity_date'] = df['maturity_date'].shift(-10) # 提前10个交易日平仓
df.iloc[-1,-1] = maturity_date[-1]
df.fillna(method='bfill',inplace=True)
df['T'] = df.apply(lambda x: ((datetime.datetime.strptime(str(x['maturity_date']),'%Y%m%d')-\
                               datetime.datetime.strptime(str(x['trade_date']),'%Y%m%d')).days+1)/365,axis=1)
df['pre_pre_close'] = df['pre_close'].shift(1)
df['rng'] = df['pre_close']/df['pre_pre_close']-1

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
    return list(df.loc[index,['exercise_price','deposit','pre_close','close']])

df['option'] = df.apply(lambda x: getOption(x['pre_close'],OpDB,x['trade_date'],x['maturity_date'],3,True),axis=1)
df[['C_exercise_price','C_deposit','C_op_pre_close','C_op_close']] = df['option'].apply(pd.Series)
df['C_VIX'] = df.apply(lambda x: BT.CalVIX(x['pre_close'],x['C_exercise_price'],x['C_op_pre_close'],x['rf'],x['T'],1),axis=1)
df['option'] = df.apply(lambda x: getOption(x['pre_close'],OpDB,x['trade_date'],x['maturity_date'],3,False),axis=1)
df[['P_exercise_price','P_deposit','P_op_pre_close','P_op_close']] = df['option'].apply(pd.Series)
df['P_VIX'] = df.apply(lambda x: BT.CalVIX(x['pre_close'],x['P_exercise_price'],x['P_op_pre_close'],x['rf'],x['T'],0),axis=1)
df.loc[df[df['C_VIX']<0.01].index,'C_VIX'] = np.nan
df.loc[df[df['P_VIX']<0.01].index,'P_VIX'] = np.nan
df.fillna(method='bfill',inplace=True)
df['pre_C_VIX'] = df['C_VIX'].shift(1)
df['C_VIX_rng'] = df['C_VIX']-df['pre_C_VIX']
df['pre_P_VIX'] = df['P_VIX'].shift(1)
df['P_VIX_rng'] = df['P_VIX']-df['pre_P_VIX']
df.dropna(axis=0,inplace=True)
df.reset_index(drop=True,inplace=True)

#%%

Signal,Price,Direction,Deposit,flag,cover,open_close,open_VIX = [],[],[],[],False,True,0,0
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
                if (df['pre_close'].loc[i]>open_close):# or (df['pre_C_VIX'].loc[i]<open_VIX):
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
                if (df['pre_close'].loc[i]<open_close):# or (df['pre_P_VIX'].loc[i]<open_VIX):
                    signal = 0
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
        if (df['rng'].loc[i]<-0.005) and (df['C_VIX_rng'].loc[i]>0.025): # 昨日涨跌幅小于-1.5%时，卖出虚值看涨期权
            signal = 1
            price = [df['C_op_close'].loc[i]]
            direction = [-1]
            deposit = [df['C_deposit'].loc[i]]
            cover,open_close,open_VIX,flag = True,df['pre_pre_close'].loc[i],df['pre_C_VIX'].loc[i],True
        elif (df['rng'].loc[i]>0.005) and (df['P_VIX_rng'].loc[i]>0.025): # 昨日涨跌幅大于1.5%时，卖出虚值看跌期权
            signal = 1
            price = [df['P_op_close'].loc[i]]
            direction = [-1]
            deposit = [df['P_deposit'].loc[i]]
            cover,open_close,open_VIX,flag = False,df['pre_pre_close'].loc[i],df['pre_P_VIX'].loc[i],True
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