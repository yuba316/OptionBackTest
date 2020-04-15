# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 08:38:39 2020

@author: yuba316
"""

import copy as c
import datetime
import math as m
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#%%

def CalVIX(S0,X,MKT,rf,T,CorP=1):
    
    BSM,sigma,up,dw = 0,0.5,1,0
    if CorP:
        count = 1
        while count<20:
            d1 = (np.log(S0/X)+(rf+sigma**2/2)*T)/(sigma*np.sqrt(T))
            d2 = d1-sigma*np.sqrt(T)
            BSM = S0*norm.cdf(d1)-X*np.exp(-rf*T)*norm.cdf(d2)
            if BSM-MKT>0: 
                up = sigma
                sigma = (sigma+dw)/2
            elif BSM-MKT<0:
                dw = sigma
                sigma = (sigma+up)/2
            else:
                break
            count = count+1
    else:
        count = 1
        while count<20:
            d1 = (np.log(S0/X)+(rf+sigma**2/2)*T)/(sigma*np.sqrt(T))
            d2 = d1-sigma*np.sqrt(T)
            BSM = X*np.exp(-rf*T)*norm.cdf(-d2)-S0*norm.cdf(-d1)
            if BSM-MKT>0: 
                up = sigma
                sigma = (sigma+dw)/2
            elif BSM-MKT<0:
                dw = sigma
                sigma = (sigma+up)/2
            else:
                break
            count = count+1
    
    return sigma

#%%

def getOpDeposit(depositDf,Point=10000):
    
    # input:
    # depositDf[DataFrame]: [trade_date, underlying_pre_close, Call_pre_close, Put_pre_close, 
    #                        Call_pre_settle, Put_pre_settle, Call_exercise, Put_exercise]
    # Point[int]: 期权价格点位比例（元/点）
    
    # output:
    # depositDf[DataFrame]: [trade_date, Call_dep, Put_dep]
    
    depositDf['Call_dep'] = Point*(depositDf['Call_pre_settle']+\
             depositDf.apply(lambda x: (0.12*x['underlying_pre_close']-x['Call_pre_close']) if \
                             (0.12*x['underlying_pre_close']-x['Call_pre_close'])>(0.07*x['underlying_pre_close']) else \
                             0.07*x['underlying_pre_close'],axis=1))
    
    depositDf['Put_dep'] = Point*depositDf.apply(lambda x: x['Put_exercise'] if \
             x['Put_exercise']<(x['Put_pre_settle']+\
              ((0.12*x['underlying_pre_close']-x['Put_pre_close']) if \
               (0.12*x['underlying_pre_close']-x['Put_pre_close'])>(0.07*x['Put_exercise']) else 0.07*x['Put_exercise'])) else \
              (x['Put_pre_settle']+((0.12*x['underlying_pre_close']-x['Put_pre_close']) if \
               (0.12*x['underlying_pre_close']-x['Put_pre_close'])>(0.07*x['Put_exercise']) else 0.07*x['Put_exercise'])),axis=1)
    
    return depositDf[['trade_date','Call_dep','Put_dep']]

#%%

def OptionBT(signalDf,depositDf,Capital=1000000,pct=0.8,Fee=2.5,Rde=0.2,Point=10000):
    
    # input:
    # signalDf[DataFrame]: [trade_date, Call_close, Put_close, Call_volume, Put_volume, signal, direction, position, pct]
    # -- signal[int]: -2: 空头平仓, -1: 多头平仓, 0: 不操作, 1: 多头开仓, 2: 空头开仓
    # -- direction[int]: 0: Put, 1: Call, 2: 双边
    # -- position[int]: -1: 减仓, 0: 不操作, 1: 加仓
    # -- pct[float]: 加仓或减仓的比例，仓位变更数量为pct*volume
    # depositDf[DataFrame]: [trade_date, Call_dep, Put_dep]
    # Capital[float]: 本金（元）
    # pct[float]: 开仓百分比
    # Fee[float]: 手续费（元/手）
    # Rde[float]: 保证金上浮比率
    # Point[int]: 期权价格点位比例（元/点）
    
    # output:
    # recordDf[DataFrame]: [trade_date, profit, deposit, signal, position, direction, log, volume, turnover, fee, opfee, capital, strategy]
    
    trade_date,open_price,profit = [signalDf['trade_date'].iloc[0]-datetime.timedelta(days=1)],[[0]],[0]
    deposit,position,direction,log,volume,turnover,fee,opfee,capital,strategy = [0],[0],[-1],['回测开始'],[[0]],[0],[0],[0],[Capital],[Capital]
    signal = [0]+list(signalDf['signal'])
    n = len(signalDf)
    for i in range(n):
        td = signalDf['trade_date'].iloc[i]
        if signalDf['signal'].iloc[i] > 0: # 开仓
            td_profit = 0 # 只要是开仓的，当日都不会有收益，因为是在收盘时开仓
            td_strategy = strategy[-1] # 开仓只有保证金和期权费的交纳，不会有累计收益的变化
            if signalDf['signal'].iloc[i] == 2: # 空头
                td_position = -1 # -1: 空头, 1: 多头
                td_fee = 0 # 义务仓手续费只会在平仓时收取
                td_opfee = 0 # 期权卖方无需缴纳权利金
                td_capital = capital[-1] # 做空只需交纳保证金，无其他资金流出
                if signalDf['direction'].iloc[i] == 2: # 卖开双边
                    td_direction = 2 # 2: 双边, 1: Call, 0: Put
                    td_log = '卖开双边'
                    td_price = signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i]
                    td_deposit = (depositDf['Call_dep'].iloc[i]+depositDf['Put_dep'].iloc[i])*(1+Rde) # 一手合约保证金
                    set_volume = signalDf['Call_volume'].iloc[i]
                elif signalDf['direction'].iloc[i] == 1: # 卖开认购
                    td_direction = 1
                    td_log = '卖开认购'
                    td_price = signalDf['Call_close'].iloc[i]
                    td_deposit = depositDf['Call_dep'].iloc[i]*(1+Rde)
                    set_volume = signalDf['Call_volume'].iloc[i]
                else: # 卖开认沽
                    td_direction = 0
                    td_log = '卖开认沽'
                    td_price = signalDf['Put_close'].iloc[i]
                    td_deposit = depositDf['Put_dep'].iloc[i]*(1+Rde)
                    set_volume = signalDf['Put_volume'].iloc[i]
                td_volume = int(pct*capital[-1]/td_deposit) # 交易数量为最大可开仓数，即所有本金用来交满保证金为止
                td_volume = td_volume if (set_volume == -1) else min(td_volume,set_volume) # 如果设置仓位的话，不得超过最大可开仓数
                td_deposit = td_deposit*td_volume # 总的保证金金额
                td_turnover = td_price*td_volume*Point # 成交额
            else: # 多头
                td_deposit = 0 # 多头不必缴纳保证金
                td_position = 1
                if signalDf['direction'].iloc[i] == 2:
                    td_direction = 2
                    td_log = '买开双边'
                    td_price = signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i]
                    set_volume = signalDf['Call_volume'].iloc[i]
                elif signalDf['direction'].iloc[i] == 1:
                    td_direction = 1
                    td_log = '买开认购'
                    td_price = signalDf['Call_close'].iloc[i]
                    set_volume = signalDf['Call_volume'].iloc[i]
                else:
                    td_direction = 0
                    td_log = '买开认沽'
                    td_price = signalDf['Put_close'].iloc[i]
                    set_volume = signalDf['Put_volume'].iloc[i]
                td_volume = int(pct*capital[-1]/(td_price*Point))
                td_fee = Fee*td_volume if (td_direction!=2) else Fee*td_volume*2 # 权利仓双边收取手续费（阿一西是真的坑……）
                if pct*capital[-1]<(td_price*td_volume*Point+td_fee*2): # 如果加上双边的手续费以后超过了开仓的资本，就应适当减仓
                    td_volume = td_volume-m.ceil((td_price*td_volume*Point+td_fee*2-pct*capital[-1])/(td_price*Point))
                td_volume = td_volume if (set_volume == -1) else min(td_volume,set_volume)
                td_fee = Fee*td_volume if (td_direction!=2) else Fee*td_volume*2
                td_opfee = td_price*td_volume*Point
                td_turnover = -1*td_opfee
                td_capital = capital[-1]+td_turnover-td_fee
            list_price = [td_price]
            list_volume = [td_volume]
        elif signalDf['signal'].iloc[i] == 0: # 不操作
            td_position = position[-1]
            td_direction = direction[-1]
            td_opfee = opfee[-1]
            td_capital = capital[-1]
            list_price = c.deepcopy(open_price[-1])
            list_volume = c.deepcopy(volume[-1])
            daily_fee = 0
            n = len(list_price) # 加仓次数
            td_cover = 0
            if td_position == -1: # 空头仓
                if td_direction == 2: # 双边
                    td_profit,td_deposit = 0,0
                    for j in range(n):
                        td_price = list_price[j]
                        td_volume = list_volume[j]
                        td_profit = td_profit+(td_price-(signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i]))*td_volume*Point
                        td_deposit = td_deposit+(depositDf['Call_dep'].iloc[i]+depositDf['Put_dep'].iloc[i])*(1+Rde)*td_volume
                        daily_fee = daily_fee+Fee*td_volume*2
                    if td_deposit>deposit[-1]: # 保证金账户不足，强制平仓
                        td_log = '强制平仓'
                        cover_volume = m.ceil((td_deposit-deposit[-1])/(depositDf['Call_dep'].iloc[i]+depositDf['Put_dep'].iloc[i]))
                        sum_volume = list_volume[0]
                        if cover_volume<=sum_volume: # 会从第一次开仓开始强制平仓，因为我们认为后续加仓代表看好行情，不应该过早平掉
                            list_volume[0] = list_volume[0]-cover_volume
                            td_turnover = -1*(signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i])*cover_volume*Point
                            td_cover = (list_price[0]-(signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i]))*cover_volume*Point
                        else:
                            for j in range(1,n,1):
                                sum_volume = sum_volume+list_volume[j]
                                if cover_volume<=sum_volume:
                                    for k in range(j-1):
                                        td_cover = td_cover+(list_price[k]-(signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i]))*list_volume[k]*Point
                                        list_volume[k] = 0
                                    td_cover = td_cover+(list_price[j]-(signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i]))*(list_volume[j]-sum_volume+cover_volume)*Point
                                    list_volume[j] = sum_volume-cover_volume
                                    td_fee = Fee*cover_volume*2 # 平仓缴纳手续费，同时平仓Call和Put，故*2
                                    td_turnover = -1*(signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i])*cover_volume*Point
                                    break
                                if j<n-1:
                                    continue
                                for k in range(j):
                                    td_cover = td_cover+(list_price[k]-(signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i]))*list_volume[k]*Point
                                    list_volume[k] = 0
                                td_fee = Fee*sum_volume*2
                                td_turnover = -1*(signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i])*sum_volume*Point
                    else:
                        td_log = '--'
                        td_fee = 0
                        td_turnover = 0
                elif td_direction == 1: # Call
                    td_profit,td_deposit = 0,0
                    for j in range(n):
                        td_price = list_price[j]
                        td_volume = list_volume[j]
                        td_profit = td_profit+(td_price-signalDf['Call_close'].iloc[i])*td_volume*Point
                        td_deposit = td_deposit+depositDf['Call_dep'].iloc[i]*(1+Rde)*td_volume
                        daily_fee = daily_fee+Fee*td_volume
                    if td_deposit>deposit[-1]:
                        td_log = '强制平仓'
                        cover_volume = m.ceil((td_deposit-deposit[-1])/depositDf['Call_dep'].iloc[i])
                        sum_volume = list_volume[0]
                        if cover_volume<=sum_volume:
                            list_volume[0] = list_volume[0]-cover_volume
                            td_turnover = -1*signalDf['Call_close'].iloc[i]*cover_volume*Point
                            td_cover = (list_price[0]-signalDf['Call_close'].iloc[i])*cover_volume*Point
                        else:
                            for j in range(1,n,1):
                                sum_volume = sum_volume+list_volume[i]
                                if cover_volume<=sum_volume:
                                    for k in range(j-1):
                                        td_cover = td_cover+(list_price[k]-signalDf['Call_close'].iloc[i])*list_volume[k]*Point
                                        list_volume[k] = 0
                                    td_cover = td_cover+(list_price[j]-signalDf['Call_close'].iloc[i])*(list_volume[j]-sum_volume+cover_volume)*Point
                                    list_volume[j] = sum_volume-cover_volume
                                    td_fee = Fee*cover_volume
                                    td_turnover = -1*signalDf['Call_close'].iloc[i]*cover_volume*Point
                                    break
                                if j<n-1:
                                    continue
                                for k in range(j):
                                    td_cover = td_cover+(list_price[k]-signalDf['Call_close'].iloc[i])*list_volume[k]*Point
                                    list_volume[k] = 0
                                td_fee = Fee*sum_volume
                                td_turnover = -1*signalDf['Call_close'].iloc[i]*sum_volume*Point
                    else:
                        td_log = '--'
                        td_fee = 0
                        td_turnover = 0
                else: # Put
                    td_profit,td_deposit = 0,0
                    for j in range(n):
                        td_price = list_price[j]
                        td_volume = list_volume[j]
                        td_profit = td_profit+(td_price-signalDf['Put_close'].iloc[i])*td_volume*Point
                        td_deposit = td_deposit+depositDf['Put_dep'].iloc[i]*(1+Rde)*td_volume
                        daily_fee = daily_fee+Fee*td_volume
                    if td_deposit>deposit[-1]:
                        td_log = '强制平仓'
                        cover_volume = m.ceil((td_deposit-deposit[-1])/depositDf['Put_dep'].iloc[i])
                        sum_volume = list_volume[0]
                        if cover_volume<=sum_volume:
                            list_volume[0] = list_volume[0]-cover_volume
                            td_turnover = -1*signalDf['Put_close'].iloc[i]*cover_volume*Point
                            td_cover = (list_price[0]-signalDf['Put_close'].iloc[i])*cover_volume*Point
                        else:
                            for j in range(1,n,1):
                                sum_volume = sum_volume+list_volume[i]
                                if cover_volume<=sum_volume:
                                    for k in range(j-1):
                                        td_cover = td_cover+(list_price[k]-signalDf['Put_close'].iloc[i])*list_volume[k]*Point
                                        list_volume[k] = 0
                                    td_cover = td_cover+(list_price[j]-signalDf['Put_close'].iloc[i])*(list_volume[j]-sum_volume+cover_volume)*Point
                                    list_volume[j] = sum_volume-cover_volume
                                    td_fee = Fee*cover_volume
                                    td_turnover = -1*signalDf['Put_close'].iloc[i]*cover_volume*Point
                                    break
                                for k in range(j):
                                    td_cover = td_cover+(list_price[k]-signalDf['Put_close'].iloc[i])*list_volume[k]*Point
                                    list_volume[k] = 0
                                td_fee = Fee*sum_volume
                                td_turnover = -1*signalDf['Put_close'].iloc[i]*sum_volume*Point
                    else:
                        td_log = '--'
                        td_fee = 0
                        td_turnover = 0
            elif td_position == 1: # 多头仓
                td_log = '--'
                td_fee = 0
                td_turnover = 0
                if td_direction == 2:
                    td_profit = 0
                    for i in range(n):
                        td_price = list_price[i]
                        td_volume = list_volume[i]
                        td_profit = td_profit+((signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i])-td_price)*td_volume*Point
                        daily_fee = daily_fee+Fee*td_volume*2
                elif td_direction == 1:
                    td_profit = 0
                    for i in range(n):
                        td_price = list_price[i]
                        td_volume = list_volume[i]
                        td_profit = td_profit+(signalDf['Call_close'].iloc[i]-td_price)*td_volume*Point
                        daily_fee = daily_fee+Fee*td_volume
                else:
                    td_profit = 0
                    for i in range(n):
                        td_price = list_price[i]
                        td_volume = list_volume[i]
                        td_profit = td_profit+(signalDf['Put_close'].iloc[i]-td_price)*td_volume*Point
                        daily_fee = daily_fee+Fee*td_volume
            else: # 平完仓后尚未做任何开仓动作
                td_profit,td_log,td_turnover,td_fee = 0,'未开仓',0,0
                continue
            td_deposit = deposit[-1] # 保证金从开仓到平仓都不会改变，通过强制平仓部分头寸来使其满足要求
            td_strategy = td_opfee+td_capital+td_profit # 后续加减仓只有手续费会影响到当日的收益，其他只改变本金、期权费和保证金账户
            td_capital = td_capital+td_cover # 若有强制平仓，则需要加入平仓所获得的金额
            list_turnover = td_turnover # 保存一下，之后还会有成交额变动
            list_fee = td_fee # 手续费也可能会再增加
            td_turnover,td_fee = 0,0
            
            if signalDf['position'].iloc[i] == 1: # 加仓
                if td_position == -1:
                    if td_direction == 2:
                        td_log = '加仓：卖开双边'
                        td_price = signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i]
                        td_deposit = (depositDf['Call_dep'].iloc[i]+depositDf['Put_dep'].iloc[i])*(1+Rde)
                    elif td_direction == 1:
                        td_log = '加仓：卖开认购'
                        td_price = signalDf['Call_close'].iloc[i]
                        td_deposit = depositDf['Call_dep'].iloc[i]*(1+Rde)
                    else:
                        td_log = '加仓：卖开认沽'
                        td_price = signalDf['Put_close'].iloc[i]
                        td_deposit = depositDf['Put_dep'].iloc[i]*(1+Rde)
                    td_volume = min(int(list_volume[0]*signalDf['pct'].iloc[i]),int((td_capital-deposit[-1])/td_deposit))
                    td_deposit = deposit[-1]+td_deposit*td_volume
                    td_turnover = td_volume*td_price*Point
                    list_price.append(td_price)
                    list_volume.append(td_volume)
                else:
                    if td_direction == 2:
                        td_log = '加仓：买开双边'
                        td_price = signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i]
                    elif td_direction == 1:
                        td_log = '加仓：买开认购'
                        td_price = signalDf['Call_close'].iloc[i]
                    else:
                        td_log = '加仓：买开认沽'
                        td_price = signalDf['Put_close'].iloc[i]
                    td_volume = int(td_capital/(td_price*Point))
                    td_fee = Fee*td_volume if (td_direction!=2) else Fee*td_volume*2
                    if td_capital<(td_price*td_volume*Point+td_fee*2):
                        td_volume = td_volume-m.ceil((td_price*td_volume*Point+td_fee*2-td_capital)/(td_price*Point))
                    td_volume = min(int(list_volume[0]*signalDf['pct'].iloc[i]),td_volume)
                    td_fee = Fee*td_volume if (td_direction!=2) else Fee*td_volume*2
                    td_opfee = td_opfee+td_price*td_volume*Point
                    td_turnover = -1*td_price*td_volume*Point
                    list_price.append(td_price)
                    list_volume.append(td_volume)
                    td_capital = td_capital+td_turnover
                    daily_fee = daily_fee+td_fee
            elif signalDf['position'].iloc[i] == -1: # 减仓
                if td_position == -1:
                    if td_direction == 2:
                        td_log = '减仓：买平双边'
                        td_volume = int(list_volume[0]*signalDf['pct'].iloc[i])
                        td_turnover = -1*(signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i])*td_volume*Point
                        td_fee = Fee*td_volume*2
                        td_capital = td_capital+(list_price[0]-(signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i]))*td_volume*Point
                    elif td_direction == 1:
                        td_log = '减仓：买平认购'
                        td_volume = int(list_volume[0]*signalDf['pct'].iloc[i])
                        td_turnover = -1*signalDf['Call_close'].iloc[i]*td_volume*Point
                        td_fee = Fee*td_volume
                        td_capital = td_capital+(list_price[0]-signalDf['Call_close'].iloc[i])*td_volume*Point
                    else:
                        td_log = '减仓：买平认沽'
                        td_volume = int(list_volume[0]*signalDf['pct'].iloc[i])
                        td_turnover = -1*signalDf['Put_close'].iloc[i]*td_volume*Point
                        td_fee = Fee*td_volume
                        td_capital = td_capital+(list_price[0]-signalDf['Put_close'].iloc[i])*td_volume*Point
                    list_volume[0] = list_volume[0]-td_volume
                else:
                    if td_direction == 2:
                        td_log = '减仓：卖平双边'
                        td_volume = int(list_volume[0]*signalDf['pct'].iloc[i])
                        td_turnover = (signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i])*td_volume*Point
                        td_fee = Fee*td_volume*2
                    elif td_direction == 1:
                        td_log = '减仓：卖平认购'
                        td_volume = int(list_volume[0]*signalDf['pct'].iloc[i])
                        td_turnover = signalDf['Call_close'].iloc[i]*td_volume*Point
                        td_fee = Fee*td_volume
                    else:
                        td_log = '减仓：卖平认沽'
                        td_volume = int(list_volume[0]*signalDf['pct'].iloc[i])
                        td_turnover = signalDf['Put_close'].iloc[i]*td_volume*Point
                        td_fee = Fee*td_volume
                    list_volume[0] = list_volume[0]-td_volume
                    td_opfee = td_opfee-list_price[0]*td_volume*Point
                    td_capital = td_capital+td_turnover
            td_turnover = list_turnover+td_turnover
            td_fee = list_fee+td_fee
            td_strategy = td_strategy-daily_fee
            td_capital = td_capital-td_fee
        else: # 平仓
            td_deposit = 0 # 平仓时清空保证金账户
            td_position = position[-1]
            td_direction = direction[-1]
            td_opfee = opfee[-1]
            td_capital = capital[-1]
            list_price = c.deepcopy(open_price[-1])
            list_volume = c.deepcopy(volume[-1])
            n = len(list_price)
            if td_position == -1:
                if td_direction == 2:
                    td_log = '买平双边'
                    td_profit = 0
                    for j in range(n):
                        td_price = list_price[j]
                        td_volume = list_volume[j]
                        td_profit = td_profit+(td_price-(signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i]))*td_volume*Point
                    td_turnover = -1*(signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i])*sum(list_volume)*Point
                    td_fee = Fee*sum(list_volume)*2
                elif td_direction == 1:
                    td_log = '买平认购'
                    td_profit = 0
                    for j in range(n):
                        td_price = list_price[j]
                        td_volume = list_volume[j]
                        td_profit = td_profit+(td_price-signalDf['Call_close'].iloc[i])*td_volume*Point
                    td_turnover = -1*signalDf['Call_close'].iloc[i]*sum(list_volume)*Point
                    td_fee = Fee*sum(list_volume)
                else:
                    td_log = '买平认沽'
                    td_profit = 0
                    for j in range(n):
                        td_price = list_price[j]
                        td_volume = list_volume[j]
                        td_profit = td_profit+(td_price-signalDf['Put_close'].iloc[i])*td_volume*Point
                    td_turnover = -1*signalDf['Put_close'].iloc[i]*sum(list_volume)*Point
                    td_fee = Fee*sum(list_volume)
            else:
                if td_direction == 2:
                    td_log = '卖平双边'
                    td_profit = 0
                    for j in range(n):
                        td_price = list_price[j]
                        td_volume = list_volume[j]
                        td_profit = td_profit+(signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i]-td_price)*td_volume*Point
                    td_turnover = (signalDf['Call_close'].iloc[i]+signalDf['Put_close'].iloc[i])*sum(list_volume)*Point
                    td_fee = Fee*sum(list_volume)*2
                elif td_direction == 1:
                    td_log = '卖平认购'
                    td_profit = 0
                    for j in range(n):
                        td_price = list_price[j]
                        td_volume = list_volume[j]
                        td_profit = td_profit+(signalDf['Call_close'].iloc[i]-td_price)*td_volume*Point
                    td_turnover = signalDf['Call_close'].iloc[i]*sum(list_volume)*Point
                    td_fee = Fee*sum(list_volume)
                else:
                    td_log = '卖平认沽'
                    td_profit = 0
                    for j in range(n):
                        td_price = list_price[j]
                        td_volume = list_volume[j]
                        td_profit = td_profit+(signalDf['Put_close'].iloc[i]-td_price)*td_volume*Point
                    td_turnover = signalDf['Put_close'].iloc[i]*sum(list_volume)*Point
                    td_fee = Fee*sum(list_volume)
            td_strategy = td_opfee+td_capital+td_profit-td_fee
            td_capital = td_strategy # 平仓后，累计收益落袋为资本
            list_price = [0] # 平仓后注意将所有指标清零
            td_position = 0
            td_direction = -1
            list_volume = [0]
            td_opfee = 0
            
        trade_date.append(td)
        open_price.append(list_price)
        profit.append(td_profit)
        deposit.append(td_deposit)
        position.append(td_position)
        direction.append(td_direction)
        log.append(td_log)
        volume.append(list_volume)
        turnover.append(td_turnover)
        fee.append(td_fee)
        opfee.append(td_opfee)
        capital.append(td_capital)
        strategy.append(td_strategy)
        
        if td_capital<=0: # 你已经破产了，兄dei~
            break
        
    recordDf = pd.DataFrame({'trade_date':trade_date, 'profit':profit, 'deposit':deposit, 'signal':signal, 'position':position, 'direction':direction, \
                'log':log, 'volume':volume, 'turnover':turnover, 'fee':fee, 'opfee':opfee, 'capital':capital, 'strategy':strategy})
    
    return recordDf

#%%

def Visualize(recordDf,benchmarkDf,title='回测结果',Underlying=True):
    
    # input:
    # recordDf[DataFrame]: OptionBT的输出结果
    # benchmarkDf[DataFrame]: 基准线行情，一般是underlying
    # Underlying[Boolean]: 是否添加underlying行情线
    
    # output:
    # stat[Series]: [total_profit, annual_profit, annual_std, Sharpe, dw_annual_std, Sortino, max_drawdown, pct_max_drawdown, Kama, win_rate, profit_loss]
    
    strategy = c.deepcopy(recordDf[['trade_date','signal','direction','capital','strategy']])
    benchmark = c.deepcopy(benchmarkDf)
    capital = strategy['capital'].iloc[0]
    S0 = benchmark['close'].iloc[0]
    
    strategy['pct_strategy'] = 100*(strategy['strategy']-capital)/capital
    benchmark['pct_benchmark'] = 100*(benchmark['close']-S0)/S0
    if Underlying:
        open_0 = list(strategy[(strategy['signal']>0)&(strategy['direction']==0)].index-1)
        open_1 = list(strategy[(strategy['signal']>0)&(strategy['direction']==1)].index-1)
        open_2 = list(strategy[(strategy['signal']>0)&(strategy['direction']==2)].index-1)
        cover = list(strategy[strategy['signal']<0].index-1)[1:]
    
    plt.figure(figsize=(12,4))
    plt.plot(strategy['trade_date'],strategy['pct_strategy'],label='策略收益%')
    plt.plot(benchmark['trade_date'],benchmark['pct_benchmark'],label='基准线收益%')
    plt.legend(loc='upper left')
    plt.title(title)
    if Underlying:
        ax = plt.twinx()
        ax.plot(benchmark['trade_date'],benchmark['close'],label='Underlying',color='limegreen')
        ax.plot(benchmark['trade_date'].iloc[open_0],benchmark['close'].iloc[open_0],'o',label='开认沽仓',color='crimson')
        ax.plot(benchmark['trade_date'].iloc[open_1],benchmark['close'].iloc[open_1],'o',label='开认购仓',color='cornflowerblue')
        ax.plot(benchmark['trade_date'].iloc[open_2],benchmark['close'].iloc[open_2],'o',label='开双边仓',color='blueviolet')
        ax.plot(benchmark['trade_date'].iloc[cover],benchmark['close'].iloc[cover],'x',label='平仓',color='gold')
        ax.legend(loc='lower right')
    
    stat = {}
    stat['total_profit'] = strategy['pct_strategy'].iloc[-1] # 累计收益
    stat['annual_profit'] = 365*stat['total_profit']/(strategy['trade_date'].iloc[-1]-strategy['trade_date'].iloc[1]).days # 年化收益率
    strategy['daily_profit'] = (strategy['strategy']-strategy['strategy'].shift(1))/strategy['strategy'].shift(1)
    strategy.fillna(0,inplace=True)
    stat['annual_std'] = strategy['daily_profit'].std()*np.sqrt(252) # 年化波动率
    stat['dw_annual_std'] = strategy[strategy['daily_profit']<0]['daily_profit'].std()*np.sqrt(252) # 年化下行波动率
    stat['Sharpe'] = 0.01*stat['annual_profit']/stat['annual_std'] # 夏普比率
    stat['Sortino'] = 0.01*stat['annual_profit']/stat['dw_annual_std'] # 索提诺比率
    profit = list(strategy['strategy'])
    index_j = np.argmax(np.maximum.accumulate(profit)-profit)
    index_i = np.argmax(profit[:index_j])
    stat['max_drawdown'] = profit[index_i]-profit[index_j] # 最大回撤
    stat['pct_max_drawdown'] = stat['max_drawdown']/capital # 最大回撤百分比
    stat['kama'] = stat['annual_profit']/stat['pct_max_drawdown'] # 卡玛比率
    
    summary = strategy[strategy['signal']!=0]
    win_rate,profit,loss = 0,[],[]
    n = len(summary)
    for i in range(n-1):
        if i%2 == 0:
            PnL = summary['strategy'].iloc[i+1] - summary['strategy'].iloc[i]
            if PnL > 0:
                win_rate = win_rate+1
                profit.append(PnL)
            else:
                loss.append(PnL)
        else:
            continue
    stat['win_rate'] = win_rate/n*2
    stat['profit_loss'] = -1*sum(profit)/sum(loss)
    
    return pd.Series(stat)