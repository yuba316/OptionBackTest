# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:23:58 2020

@author: yuba316
"""

import sys
sys.path.append(r'D:\work\back_test_system')
import DataBase as DB
import BackTest_2 as BT

import copy as c
import math as m
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd

import datetime

#%% 计算SABR

def CalSabr(F,K,T,arg,beta=0):
    
    alpha,rho,vega = arg[0],arg[1],arg[2]
    z = (vega/alpha)*(F*K)**((1-beta)/2)*np.log(F/K)
    X = np.log((np.sqrt(1-2*rho*z+z**2)+z-rho)/(1-rho))
    a = ((((1-beta)*alpha)**2/(24*(F*K)**(1-beta))+\
          rho*beta*vega*alpha/(4*(F*K)**((1-beta)/2))+\
          (2-3*rho**2)*vega**2/24)*T+1)*alpha
    b = ((F*K)**((1-beta)/2))*(1+((1-beta)*np.log(F/K))**2/24+((1-beta)*np.log(F/K))**4/1920)
    
    return a/b*z/X

def fun(args):
    
    VIX,F,K,T,beta = args
    v = lambda x: abs(VIX-CalSabr(F,K,T,x,beta))
    
    return v


def con(args):
    
    x1min,x1max,x2min,x2max,x3min,x3max = args
    cons = ({'type':'ineq','fun':lambda x: x[0]-x1min},\
             {'type':'ineq','fun':lambda x: x1max-x[0]},\
             {'type':'ineq','fun':lambda x: x[1]-x2min},\
             {'type':'ineq','fun':lambda x: x2max-x[1]},\
             {'type':'ineq','fun':lambda x: x[2]-x3min},\
             {'type':'ineq','fun':lambda x: x3max-x[2]})
    
    return cons

#%% 计算delta

def CalDelta(S0,X,sigma,rf,T,CorP=1):
    
    d1 = (np.log(S0/X)+(rf+sigma**2/2)*T)/(sigma*np.sqrt(T))
    if CorP:
        delta = norm.cdf(d1)
    else:
        delta = 1-norm.cdf(d1)
    
    return delta

#%% 寻找每日可开仓的期权合约

df = DB.getUnderlying(end_date='20200410')[['trade_date','close','pre_close']]
df['sigma'] = df['pre_close'].rolling(window=21).std()
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

#%% 自己定义今天想要什么样的期权

OpDB = pd.read_csv(r'D:\work\back_test_system\DataBase\Option\OP510050.csv')
OpDB[['trade_date','maturity_date','list_date']] = OpDB[['trade_date','maturity_date','list_date']].applymap(str)
def getTdOp(pre_close,sigma,OpDB,td,maturity_date,rf):
    
    df = c.deepcopy(OpDB[(OpDB['trade_date']==td)&(OpDB['maturity_date']==maturity_date)&(OpDB['pre_close']>0.01)])
    df['underlying_pre_close'] = pre_close
    df['rf'] = rf
    T = ((datetime.datetime.strptime(maturity_date,'%Y%m%d')-datetime.datetime.strptime(td,'%Y%m%d')).days+1)/365
    df['T'] = T
    if len(df)==0:
        return []
    df['call_put'] = df['call_put'].apply(lambda x: 1 if x=='C' else 0)
    df['VIX'] = df.apply(lambda x: BT.CalVIX(x['underlying_pre_close'],x['exercise_price'],x['pre_close'],x['rf'],x['T'],x['call_put']),axis=1)
    df['delta'] = df.apply(lambda x: CalDelta(x['underlying_pre_close'],x['exercise_price'],x['VIX'],x['rf'],x['T'],x['call_put']),axis=1)
    df = df[(df['delta']>0.1)&(df['delta']<0.9)]
    df.reset_index(drop=True,inplace=True)
    
    n = len(df)
    if n==0:
        return []
    arg = []
    x0 = np.asarray((0.5,0.5,0.5)) # alpha, rho, vega的初始值设定
    arg_1 = (0,1,-1,1,0,1) # 三者的取值范围
    cons = con(arg_1)
    for i in range(n):
        VIX,F,K,T,beta = sigma,df['underlying_pre_close'].iloc[i],df['exercise_price'].iloc[i],df['T'].iloc[i],0
        arg_0 = (VIX,F,K,T,beta)
        res = minimize(fun(arg_0),x0,method='SLSQP',constraints=cons)
        arg.append(res.x)
    
    df['arg'] = arg
    df['SABR'] = df.apply(lambda x: CalSabr(x['underlying_pre_close'],x['exercise_price'],x['T'],x['arg']),axis=1)
    df['dis'] = df['VIX']-df['SABR']
    df.dropna(inplace=True)
    if len(df)==0:
        return []
    
    option_code = []
    call,put = c.deepcopy(df[df['call_put']==0][['ts_code','delta','dis']]),c.deepcopy(df[df['call_put']==1][['ts_code','delta','dis']])
    if len(call)==0:
        option_code.append(put[put['dis']==put['dis'].max()]['ts_code'].iloc[0])
        option_code.append(put[put['dis']==put['dis'].min()]['ts_code'].iloc[0])
        option_code.append((put[put['dis']==put['dis'].min()]['delta'].iloc[0])/(put[put['dis']==put['dis'].max()]['delta'].iloc[0]))
        option_code.append(put['dis'].max()-put['dis'].min())
    elif len(put)==0:
        option_code.append(call[call['dis']==call['dis'].max()]['ts_code'].iloc[0])
        option_code.append(call[call['dis']==call['dis'].min()]['ts_code'].iloc[0])
        option_code.append((call[call['dis']==call['dis'].min()]['delta'].iloc[0])/(call[call['dis']==call['dis'].max()]['delta'].iloc[0]))
        option_code.append(call['dis'].max()-call['dis'].min())
    elif (call['dis'].max()-call['dis'].min())>=(put['dis'].max()-put['dis'].min()):
        option_code.append(call[call['dis']==call['dis'].max()]['ts_code'].iloc[0])
        option_code.append(call[call['dis']==call['dis'].min()]['ts_code'].iloc[0])
        option_code.append((call[call['dis']==call['dis'].min()]['delta'].iloc[0])/(call[call['dis']==call['dis'].max()]['delta'].iloc[0]))
        option_code.append(call['dis'].max()-call['dis'].min())
    else:
        option_code.append(put[put['dis']==put['dis'].max()]['ts_code'].iloc[0])
        option_code.append(put[put['dis']==put['dis'].min()]['ts_code'].iloc[0])
        option_code.append((put[put['dis']==put['dis'].min()]['delta'].iloc[0])/(put[put['dis']==put['dis'].max()]['delta'].iloc[0]))
        option_code.append(put['dis'].max()-put['dis'].min())
    
    return option_code

df['code'] = df.apply(lambda x: getTdOp(x['pre_close'],x['sigma'],OpDB,x['trade_date'],x['maturity_date'],x['rf']),axis=1)

#%% 设置买入卖出阈值

df[['sell','buy','delta','dis']] = df['code'].apply(pd.Series)
signal,dis,sell,buy,n = [1],df['dis'].iloc[0],df['sell'].iloc[0],df['buy'].iloc[0],len(df)
for i in range(1,n,1):
    if (df['sell'].iloc[i]!=sell)and(df['buy'].iloc[i]!=buy):
        signal.append(2)
        dis = df['dis'].iloc[i]
        sell = df['sell'].iloc[i]
        buy = df['buy'].iloc[i]
    elif df['dis'].iloc[i]>dis:
        signal.append(3) # 加仓
        dis = df['dis'].iloc[i]
    else:
        signal.append(0)
df['signal'] = signal
df.iloc[-1,-1] = -1

#%% 按回测函数的要求生成信号表

trade_date,signal,code,direction,volume,position,pct = [df['trade_date'].iloc[0]],[df['signal'].iloc[0]],[],[],[],[[0,0]],[[0,0]]
if signal[0]==1:
    code.append([df['sell'].iloc[0],df['buy'].iloc[0]])
    direction.append([-1,1])
    volume.append([int(100*df['delta'].iloc[0]),100])
else:
    code.append([])
    direction.append([])
    volume.append([])
n = len(df)
for i in range(1,n,1):
    trade_date.append(df['trade_date'].iloc[i])
    if df['signal'].iloc[i]==1:
        signal.append(1)
        code.append([df['sell'].iloc[i],df['buy'].iloc[i]])
        direction.append([-1,1])
        volume.append([int(100*df['delta'].iloc[i]),100])
        position.append([0,0])
        pct.append([0,0])
    elif df['signal'].iloc[i]==0:
        signal.append(0)
        code.append(code[-1])
        direction.append(direction[-1])
        volume.append(volume[-1])
        position.append([0,0])
        pct.append([0,0])
    elif df['signal'].iloc[i]==3:
        signal.append(0)
        code.append(code[-1])
        direction.append(direction[-1])
        volume.append(volume[-1])
        position.append([1,1])
        pct.append([0.2,0.2])
    elif df['signal'].iloc[i]==4:
        signal.append(0)
        code.append(code[-1])
        direction.append(direction[-1])
        volume.append(volume[-1])
        position.append([-1,-1])
        pct.append([0.5,0.5])
    elif df['signal'].iloc[i]==2:
        signal.append(-1)
        code.append(code[-1])
        direction.append(direction[-1])
        volume.append(volume[-1])
        position.append([0,0])
        pct.append([0,0])
        trade_date.append(trade_date[-1])
        signal.append(1)
        code.append([df['sell'].iloc[i],df['buy'].iloc[i]])
        direction.append([-1,1])
        volume.append([int(100*df['delta'].iloc[0]),100])
        position.append([0,0])
        pct.append([0,0])
    else:
        signal.append(-1)
        code.append(code[-1])
        direction.append(direction[-1])
        volume.append(volume[-1])
        position.append([0,0])
        pct.append([0,0])
signalDf = pd.DataFrame({'trade_date':trade_date,'signal':signal,'code':code,\
                        'direction':direction,'volume':volume,'position':position,'pct':pct})
signalDf[['ts_code','buy']] = signalDf['code'].apply(pd.Series)
signalDf = pd.merge(signalDf,OpDB[['trade_date','ts_code','close','deposit']],how='left',on=['trade_date','ts_code'])
signalDf.rename(columns={'ts_code':'sell','buy':'ts_code','close':'sell_close','deposit':'sell_deposit'},inplace=True)
signalDf = pd.merge(signalDf,OpDB[['trade_date','ts_code','close','deposit']],how='left',on=['trade_date','ts_code'])
signalDf['price'] = signalDf.apply(lambda x: np.array([x['sell_close'],x['close']]),axis=1)
signalDf['deposit'] = signalDf.apply(lambda x: np.array([x['sell_deposit'],x['deposit']]),axis=1)
signalDf.drop(['sell','ts_code','sell_close','close','sell_deposit'],axis=1,inplace=True)
signalDf['trade_date'] = signalDf['trade_date'].apply(str)
signalDf['trade_date'] = signalDf['trade_date'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m%d'))

#%% 回测

recordDf = BT.OptionBT(signalDf)
benchmarkDf = c.deepcopy(df[['trade_date','close']])
benchmarkDf['trade_date'] = benchmarkDf['trade_date'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m%d'))
benchmarkDf = benchmarkDf[(benchmarkDf['trade_date']>recordDf['trade_date'].iloc[0])&(benchmarkDf['trade_date']<=recordDf['trade_date'].iloc[-1])]
benchmarkDf.reset_index(drop=True,inplace=True)
stat = BT.Visualize(recordDf,benchmarkDf)