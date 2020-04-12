# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 08:20:15 2020

@author: yuba316
"""

import sys
sys.path.append(r'D:\work\back_test_system')
import DataBase as DB
import BackTest as BT

import copy as c
import math as m
import numpy as np
import pandas as pd

import datetime

#%% 获得基于标的物所生成的交易信号

ma,devU,devD,observer = 40,0.03,0.03,5
flag = 1/observer*m.ceil(observer/2)
df = DB.getUnderlying()[['trade_date','close','pre_close']]
df['ma'] = df['pre_close'].rolling(window=ma).mean()
df['signal'] = df.apply(lambda x: 1 if x['pre_close']>=x['ma']*(1+devU) else (-1 if x['pre_close']<=x['ma']*(1-devD) else 0),axis=1)
df['observer'] = df['signal'].rolling(window=observer).mean()
df['observer'] = df['observer'].apply(lambda x: 1 if x>=flag else (-1 if x<=-1*flag else 0))
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

#%% 获取每月的期权到期时间

start_date = df['trade_date'].iloc[0]
end_date = df['trade_date'].iloc[-2]
option_basic = DB.getOpBasic()
maturity_date = list(option_basic[(option_basic['maturity_date']>start_date)&(option_basic['maturity_date']<end_date)]['maturity_date'].unique())
maturity_date.append(list(option_basic[option_basic['maturity_date']>maturity_date[-1]]['maturity_date'].unique())[0])

#%% 生成开平仓信号

option_maturity_date,order,direction = [],[],[]
n = len(df)

option_maturity_date.append(maturity_date[0])
order.append(2)
direction.append(df['signal'].iloc[0])

i=1
while(i<n-1):
    if (i+3<n-1) and df['trade_date'].iloc[i+3] in maturity_date: # 还有4天到期时开仓
        option_maturity_date.append(maturity_date[maturity_date.index(df['trade_date'].iloc[i+3])+1])
        order.append(2)
        direction.append(df['signal'].iloc[i])
    elif (i+4<n-1) and df['trade_date'].iloc[i+4] in maturity_date: # 还有5天到期时平仓
        option_maturity_date.append(option_maturity_date[-1])
        order.append(-2)
        direction.append(direction[-1])
    else:
        option_maturity_date.append(option_maturity_date[-1])
        direction.append(direction[-1])
        last_cover_date = maturity_date[maturity_date.index(option_maturity_date[-1])-1]
        if (i>4) and (i+6<n-1) and (df['trade_date'].iloc[i-4] > last_cover_date) and (df['trade_date'].iloc[i+5] < option_maturity_date[-1])\
        and (direction[-1]!=df['observer'].iloc[i]): # 开仓已过去5天，且距离平仓还有2天时间，开启监视哨
            order.append(-2)
            option_maturity_date.append(option_maturity_date[-1])
            order.append(2)
            direction.append(df['signal'].iloc[i])
            i = i+2
            continue
        order.append(0)
    i = i+1

option_maturity_date.append(option_maturity_date[-1])
order.append(-2)
direction.append(direction[-1])
df['maturity_date'] = option_maturity_date
df['order'] = order
df['direction'] = direction

#%% 合并期权合约行情

n = len(df)
df['call'],df['put'] = np.nan,np.nan
for i in range(n):
    if df['order'].iloc[i]==2:
        option = c.deepcopy(option_basic[(option_basic['maturity_date']==df['maturity_date'].iloc[i])&\
                                         (option_basic['list_date']<=df['trade_date'].iloc[i])])
        option['distance'] = abs(option['exercise_price']-df['pre_close'].iloc[i])
        call = option[option['call_put']=='C']
        call_distance = list(call['distance'])
        call_index = min(call_distance.index(min(call_distance))+1,len(call)-1)
        df['call'].iloc[i] = call['ts_code'].iloc[call_index]
        put = option[option['call_put']=='P']
        put_distance = list(put['distance'])
        put_index = max(put_distance.index(min(put_distance))-1,0)
        df['put'].iloc[i] = put['ts_code'].iloc[put_index]
df.fillna(method='ffill',inplace=True)
code = list(df['call'].unique())+list(df['put'].unique())
OpDB = pd.read_csv(r'D:\work\back_test_system\DataBase\Option\OP510050.csv')
OpDB = OpDB[OpDB['ts_code'].apply(lambda x: x in code)]
OpDB['trade_date'] = OpDB['trade_date'].apply(str)

#%% 拆分成signalDf、depositDf和vixDf

# signalDf[DataFrame]: [trade_date, Call_close, Put_close, Call_volume, Put_volume, signal, direction, position, pct]
signalDf = c.deepcopy(df[['trade_date','close','pre_close','order','direction','call','put','maturity_date']])
signalDf.rename(columns={'close':'underlying_close','pre_close':'underlying_pre_close','order':'signal'},inplace=True)
signalDf['direction'] = signalDf['direction'].apply(lambda x: 0 if x==1 else(1 if x==-1 else 2))

signalDf.rename(columns={'call':'ts_code'},inplace=True)
signalDf = pd.merge(signalDf,OpDB[['ts_code','trade_date','close','pre_close','pre_settle','exercise_price']],how='left',on=['ts_code','trade_date'])
signalDf.rename(columns={'close':'Call_close','pre_close':'Call_pre_close','pre_settle':'Call_pre_settle','exercise_price':'Call_exercise'},inplace=True)
signalDf.rename(columns={'ts_code':'call'},inplace=True)

signalDf.rename(columns={'put':'ts_code'},inplace=True)
signalDf = pd.merge(signalDf,OpDB[['ts_code','trade_date','close','pre_close','pre_settle','exercise_price']],how='left',on=['ts_code','trade_date'])
signalDf.rename(columns={'close':'Put_close','pre_close':'Put_pre_close','pre_settle':'Put_pre_settle','exercise_price':'Put_exercise'},inplace=True)
signalDf.rename(columns={'ts_code':'put'},inplace=True)

# depositDf[DataFrame]: [trade_date, underlying_pre_close, Call_pre_close, Put_pre_close, Call_pre_settle, Put_pre_settle, Call_exercise, Put_exercise]
depositDf = c.deepcopy(signalDf['trade_date, underlying_pre_close, Call_pre_close, Put_pre_close, Call_pre_settle, Put_pre_settle, Call_exercise, Put_exercise'.split(', ')])
vixDf = c.deepcopy(signalDf[['trade_date','underlying_close','Call_close','Put_close','Call_exercise','Put_exercise','maturity_date']])
signalDf = signalDf[['trade_date','signal','direction','Call_close','Put_close']]

#%% 计算隐含波动率，准备加减仓

rfDf = DB.getRf(vixDf['trade_date'].iloc[0],vixDf['trade_date'].iloc[-1])
vixDf = pd.merge(vixDf,rfDf,how='left',on='trade_date')
vixDf['T'] = vixDf.apply(lambda x: ((datetime.datetime.strptime(x['maturity_date'],'%Y%m%d')-\
     datetime.datetime.strptime(x['trade_date'],'%Y%m%d')).days+1)/365,axis=1)

vixDf['CorP'] = 1
vixDf['Call_vix'] = vixDf.apply(lambda x: BT.CalVIX(x['underlying_close'],x['Call_exercise'],x['Call_close'],x['rf'],x['T'],x['CorP']),axis=1)
vixDf['CorP'] = 0
vixDf['Put_vix'] = vixDf.apply(lambda x: BT.CalVIX(x['underlying_close'],x['Put_exercise'],x['Put_close'],x['rf'],x['T'],x['CorP']),axis=1)

vixDf['Call_rng'] = vixDf['Call_vix']-vixDf['Call_vix'].shift(1)
vixDf['Put_rng'] = vixDf['Put_vix']-vixDf['Put_vix'].shift(1)
vixDf['Call_rng'] = vixDf['Call_rng'].shift(1)
vixDf['Put_rng'] = vixDf['Put_rng'].shift(1)
vixDf.fillna(0,inplace=True)

signalDf = pd.merge(signalDf,vixDf[['trade_date','Call_rng','Put_rng']],how='left',on='trade_date')

#%% 加减仓

signalDf['position'] = 0
n = len(signalDf)
flagOp,flagCl = True,True
for i in range(n):
    if signalDf['signal'].iloc[i]==0:
        if flagOp and ((signalDf['direction'].iloc[i]==0 and signalDf['Put_rng'].iloc[i]>=0.05) or \
        (signalDf['direction'].iloc[i]==1 and signalDf['Call_rng'].iloc[i]>=0.05) or \
        (signalDf['direction'].iloc[i]==2 and signalDf['Call_rng'].iloc[i]>=0.05 and signalDf['Put_rng'].iloc[i]>=0.05)):
            signalDf['position'].iloc[i] = 1
            flagOp = False
        elif flagCl and ((signalDf['direction'].iloc[i]==0 and signalDf['Put_rng'].iloc[i]<=-0.05) or \
        (signalDf['direction'].iloc[i]==1 and signalDf['Call_rng'].iloc[i]<=-0.05) or \
        (signalDf['direction'].iloc[i]==2 and (signalDf['Call_rng'].iloc[i]<=-0.05 or signalDf['Put_rng'].iloc[i]<=-0.05))):
            signalDf['position'].iloc[i] = -1
            flagCl = False
    elif signalDf['signal'].iloc[i]==2:
        flagOp,flagCl = True,True
signalDf['pct'] = 0.1
signalDf['Call_volume'] = -1
signalDf['Put_volume'] = -1
signalDf.drop(['Call_rng','Put_rng'],axis=1,inplace=True)

#%% 回测

signalDf['trade_date'] = signalDf['trade_date'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m%d'))
depositDf['trade_date'] = depositDf['trade_date'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m%d'))
depositDf = BT.getOpDeposit(depositDf,10000)
recordDf = BT.OptionBT(signalDf,depositDf)
benchmarkDf = c.deepcopy(df[['trade_date','close']])
benchmarkDf['trade_date'] = benchmarkDf['trade_date'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m%d'))
stat = BT.Visualize(recordDf,benchmarkDf)