# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:35:13 2020

@author: yuba316
"""

import os
import copy as c
import math as m
import numpy as np
import pandas as pd

import time as t
import datetime

import tushare as ts
pro = ts.pro_api('da949c80ceb5513dcc45b50ba0b0dec1bc518132101bec0dfb19da56')

#%%

def CalOpDeposit(underlying_pre_close,option_pre_close,option_pre_settle,option_exercise,CorP=1,Point=10000):
    
    if CorP:
        deposit = Point*(option_pre_settle+max(0.12*underlying_pre_close-option_pre_close,0.07*underlying_pre_close))
    else:
        deposit = Point*min(option_exercise,option_pre_settle+max(0.12*underlying_pre_close-option_pre_close,0.07*option_exercise))
    
    return deposit


#%%

def getOpBasic(exchange='SSE',underlying='OP510050.SH',start_date=None,end_date=None):
    
    option_basic = pro.opt_basic(exchange=exchange,fields='ts_code,opt_code,call_put,exercise_price,maturity_date,list_date')
    option_basic = option_basic[option_basic['opt_code']==underlying]
    
    if start_date is not None:
        if end_date is not None:
            option_basic = option_basic[(option_basic['maturity_date']>=start_date)&(option_basic['maturity_date']<=end_date)]
        else:
            option_basic = option_basic[option_basic['maturity_date']>=start_date]
    else:
        if end_date is not None:
            option_basic = option_basic[option_basic['maturity_date']<=end_date]
    
    option_basic.sort_values(by=['maturity_date','exercise_price','call_put'],inplace=True)
    option_basic.reset_index(drop=True,inplace=True)
    option_basic.drop(['opt_code'],axis=1,inplace=True)
    
    return option_basic

def getOpDB(option_basic,path):
    
    if not os.path.exists(path):
        maturity_date = option_basic['maturity_date'].unique()
        OpDB = pd.DataFrame(columns='ts_code,trade_date,pre_settle,pre_close,open,high,low,close,settle,vol,amount,oi,exercise_price'.split(','))
    else:
        OpDB = pd.read_csv(path)
        maturity_date = option_basic[option_basic['maturity_date']>=OpDB['trade_date'].iloc[-1]]['maturity_date'].unique()
    
    count = 0
    ts_code = option_basic[option_basic['maturity_date'].apply(lambda x: x in maturity_date)]['ts_code']
    for i in ts_code:
        count = count+1
        if count%150==0:
            t.sleep(60)
        df = pro.opt_daily(ts_code=i,fields='ts_code,trade_date,pre_settle,pre_close,open,high,low,close,settle,vol,amount,oi')
        df['exercise_price'] = option_basic[option_basic['ts_code']==i]['exercise_price'].iloc[0]
        df['call_put'] = option_basic[option_basic['ts_code']==i]['call_put'].iloc[0]
        df['maturity_date'] = option_basic[option_basic['ts_code']==i]['maturity_date'].iloc[0]
        df['list_date'] = option_basic[option_basic['ts_code']==i]['list_date'].iloc[0]
        OpDB = pd.concat([OpDB,df],ignore_index=True,sort=False)
    OpDB.drop_duplicates(['ts_code','trade_date'],inplace=True)
    OpDB.sort_values(by=['ts_code','trade_date'],inplace=True)
    OpDB.reset_index(drop=True,inplace=True)
    OpDB.to_csv(path,index=False)
    
    return
'''
def getOpDB(option_basic,path,start_date=None):
    
    count = 0
    if start_date is None:
        maturity_date = option_basic['maturity_date'].unique()
    else: # 若设定了更新日期，则仅更新该日以后到期的期权数据
        maturity_date = option_basic[option_basic['maturity_date']>=start_date]['maturity_date'].unique()
    
    for i in maturity_date:
        ts_code = option_basic[option_basic['maturity_date']==i]['ts_code']
        wb = pd.ExcelWriter(path+"\\"+i+".xlsx")
        for j in ts_code:
            count = count+1
            if count%150==0:
                t.sleep(60)
            df = pro.opt_daily(ts_code=j,fields='trade_date,pre_settle,pre_close,open,high,low,close,settle,vol,amount,oi')
            df['exercise_price'] = option_basic[option_basic['ts_code']==j]['exercise_price'].iloc[0]
            df['call_put'] = option_basic[option_basic['ts_code']==i]['call_put'].iloc[0]
            df['maturity_date'] = option_basic[option_basic['ts_code']==i]['maturity_date'].iloc[0]
            df['list_date'] = option_basic[option_basic['ts_code']==i]['list_date'].iloc[0]
            df.sort_values(by='trade_date',inplace=True)
            df.reset_index(drop=True,inplace=True)
            df.to_excel(wb,sheet_name=j,index=False)
        wb.save()
    
    return
'''
#%%

def getUnderlying(underlying='510050.SH',start_date='20150209',end_date=None):
    
    start_date = datetime.datetime.strptime(start_date,'%Y%m%d')
    if end_date is None:
        end_date = datetime.datetime.now()
    else:
        end_date = datetime.datetime.strptime(end_date,'%Y%m%d')
    days = int((end_date-start_date).days/1000)
    stop_point = [datetime.datetime.strftime(start_date,'%Y%m%d')]
    for i in range(days):
        stop_point.append(datetime.datetime.strftime(start_date+datetime.timedelta(days=(i+1)*1000),'%Y%m%d'))
    stop_point.append(datetime.datetime.strftime(end_date,'%Y%m%d'))
    
    fields = 'trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount'
    Underlying = pd.DataFrame(columns=fields.split(","))
    for j in range(days+1):
        df = pro.fund_daily(ts_code=underlying,start_date=stop_point[j],end_date=stop_point[j+1],fields=fields)
        Underlying = pd.concat([Underlying,df],ignore_index=True,sort=False)
    Underlying.sort_values(by='trade_date',inplace=True)
    Underlying.drop_duplicates(['trade_date'],inplace=True)
    Underlying.reset_index(drop=True,inplace=True)
    
    return Underlying

def getRf(start_date='20150223',end_date=None,field='date,1y'):
    
    start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
    if end_date is None:
        end_date = datetime.datetime.now()
    else:
        end_date = datetime.datetime.strptime(end_date,'%Y%m%d')
    days = int((end_date-start_date).days/2000)
    stop_point = [datetime.datetime.strftime(start_date, '%Y%m%d')]
    for i in range(days):
        stop_point.append(datetime.datetime.strftime(start_date+datetime.timedelta(days=(i+1)*2000), '%Y%m%d'))
    stop_point.append(datetime.datetime.strftime(end_date, '%Y%m%d'))
    
    rf = pd.DataFrame(columns=field.split(","))
    for j in range(days+1):
        df = pro.shibor(start_date=stop_point[j], end_date=stop_point[j+1], fields=field)
        rf = pd.concat([rf,df], ignore_index=True, sort=False)
    rf = rf.rename(columns={'date':'trade_date','1y':'rf'})
    rf = rf.sort_values(by='trade_date')
    rf = rf.drop_duplicates(['trade_date'])
    rf = rf.reset_index(drop=True)
    rf['rf'] = rf['rf']*0.01
    
    return rf

#%%

def getFtBasic(exchange=None,ft_type='1',start_date=None,end_date=None):
    
    if exchange is None:
        exchange = ['DCE','CZCE','SHFE']
        future_basic = pd.DataFrame(columns=['ts_code','delist_date'])
        for i in exchange:
            df = pro.fut_basic(exchange=i,fut_type=ft_type,fields='ts_code,delist_date')
            future_basic = pd.concat([future_basic,df],ignore_index=True,sort=False)
    else:
        future_basic = pro.fut_basic(exchange=i,fut_type=ft_type,fields='ts_code,delist_date')
    
    if start_date is not None:
        if end_date is not None:
            future_basic = future_basic[(future_basic['delist_date']>=start_date)&(future_basic['delist_date']<=end_date)]
        else:
            future_basic = future_basic[future_basic['delist_date']>=start_date]
    else:
        if end_date is not None:
            future_basic = future_basic[future_basic['delist_date']<=end_date]
    
    future_basic.sort_values(by=['delist_date'],inplace=True)
    future_basic.reset_index(drop=True,inplace=True)
    future_basic['category'] = future_basic['ts_code'].str.extract("([A-Z]+)[0-9]")
    
    return future_basic

def getFtDB(future_basic,path,start_date=None):
    
    #count = 0
    if start_date is None:
        updateDf = c.deepcopy(future_basic)
    else:
        updateDf = c.deepcopy(future_basic[future_basic['delist_date']>=start_date])
    category = updateDf['category'].unique()
    
    for i in category:
        ts_code = updateDf[updateDf['category']==i]['ts_code']
        if not os.path.exists(path+"\\"+i):
            os.makedirs(path+"\\"+i)
        for j in ts_code:
            #count = count+1
            #if count%150==0:
            #    t.sleep(60)
            df = pro.fut_daily(ts_code=j,fields='trade_date,pre_settle,pre_close,open,high,low,close,settle,vol,amount,oi')
            df.sort_values(by='trade_date',inplace=True)
            df.reset_index(drop=True,inplace=True)
            df.to_excel(path+"\\"+i+"\\"+j+".xlsx",index=False)
    
    return

#%% 上次更新：20200410
'''
option_basic = getOpBasic()
getOpDB(option_basic,r"D:\work\back_test_system\DataBase\Option\OP510050.csv")
underlying = getUnderlying(end_date='20200410')
future_basic = getFtBasic()
getFtDB(future_basic,r"D:\work\back_test_system\DataBase\Future")
'''

#%%
'''
OpDB = pd.read_csv(r'D:\work\back_test_system\DataBase\Option\OP510050.csv')
OpDB[['trade_date','maturity_date','list_date']] = OpDB[['trade_date','maturity_date','list_date']].applymap(str)
df = c.deepcopy(underlying[['trade_date','pre_close']])
df.rename(columns={'pre_close':'underlying_pre_close'},inplace=True)
OpDB = pd.merge(OpDB,df[['trade_date','underlying_pre_close']],how='left',on='trade_date')
OpDB['CorP'] = OpDB['call_put'].apply(lambda x: 1 if x=='C' else 0)
OpDB['deposit'] = OpDB.apply(lambda x: CalOpDeposit(x['underlying_pre_close'],x['pre_close'],x['pre_settle'],x['exercise_price'],x['CorP']),axis=1)
OpDB.drop(['underlying_pre_close','CorP'],axis=1,inplace=True)
OpDB.to_csv(r'D:\work\back_test_system\DataBase\Option\OP510050.csv',index=False)
'''