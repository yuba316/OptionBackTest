# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 08:59:31 2020

@author: yuba316
"""

import copy as co
import datetime
import math as ma
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

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

def OptionBT(signalDf,Capital=1000000,pct=0.8,Fee=2.5,Rde=0.2,Point=10000):
    
    # input:
    # signalDf[DataFrame]: [trade_date, signal, price, direction, volume, deposit, position, pct]
    # -- signal[int]: -1: 平仓, 0: 不操作, 1: 开仓
    # -- price[list]: 按顺序存储的合约收盘价
    # -- direction[list]: 按顺序存储多空头仓位信息, 1: 多头, -1: 空头, 0: 不操作
    # -- volume[list]: 按顺序存储的合约交易手数, -1: 按最大可开仓数平均开仓, 0: 本次开仓不对该合约进行任何操作
    # -- deposit[list]: 按顺序存储的合约每日保证金
    # -- position[list]: -1: 减仓, 0: 不操作, 1: 加仓
    # -- pct[list]: 加仓或减仓的比例，仓位变更数量为pct*volume
    # Capital[float]: 本金（元）
    # pct[float]: 开仓百分比
    # Fee[float]: 手续费（元/手）
    # Rde[float]: 保证金上浮比率
    # Point[int]: 期权价格点位比例（元/点）
    
    # output:
    # recordDf[DataFrame]: [trade_date, profit, price, volume, direction, deposit, fee, opfee, log, capital, strategy]
    
    trade_date,price,volume,direction = [signalDf['trade_date'].iloc[0]-datetime.timedelta(days=1)],[[[0]]],[[[0]]],[np.array([0])]
    profit,deposit,fee,opfee,log,capital,strategy = [0],[np.array([0])],[0],[0],['开始回测'],[Capital],[Capital]
    signal = [0]+list(signalDf['signal'])
    n,flag = len(signalDf),False
    for i in range(n):
        td = signalDf['trade_date'].iloc[i]
        if signalDf['signal'].iloc[i]==1: # 开仓
            flag = True
            td_log = '所有仓位开仓'
            td_profit = 0 # 开仓不会产生任何收益，因为是在收盘时才开仓
            original_price = signalDf['price'].iloc[i] # 开仓时的价格
            original_volume = signalDf['volume'].iloc[i] # 初始仓位
            td_direction = signalDf['direction'].iloc[i] # 获取今日开仓的仓位方向，是多头还是空头？
            original_deposit = signalDf['deposit'].iloc[i] # 每张合约的保证金
            td_price = []
            m = len(original_price) # 总共开了多少个仓位
            for j in range(m):
                td_price.append([original_price[j]]) # 每个仓位都用一个list装起来，方便后续的加仓再操作
            td_deposit,total_fee,td_fee,td_opfee = np.array([]),np.array([]),np.array([]),np.array([])
            total_deposit,set_deposit,other_deposit,td_volume_cp,set_volume_pct = 0,np.array([]),np.array([]),np.array([]),np.array([]) # 用来存储有设定开仓数量的合约，将会优先开仓
            total_volume = sum(k for k in original_volume if k>0)
            for j in range(m):
                if td_direction[j]>0: # 多头仓
                    td_deposit = np.append(td_deposit,original_price[j]*Point+2*Fee) # 多头仓保存的是每一张期权的期权费和手续费
                    total_fee = np.append(total_fee,2*Fee) # 多头需双边缴纳保证金，开仓时尽量保证本金足够偿还平仓时产生的费用
                    td_fee = np.append(td_fee,Fee)
                    td_opfee = np.append(td_opfee,original_price[j]*Point) # 买入期权需要缴纳期权费，算作是资金的支出
                elif td_direction[j]<0: # 空头
                    td_deposit = np.append(td_deposit,original_deposit[j]+Fee) # 空头则保存保证金
                    total_fee = np.append(total_fee,Fee)
                    td_fee = np.append(td_fee,0)
                    td_opfee = np.append(td_opfee,0)
                else: # 本次开仓不对该合约进行任何操作
                    td_deposit = np.append(td_deposit,0)
                    total_fee = np.append(total_fee,0)
                    td_fee = np.append(td_fee,0)
                    td_opfee = np.append(td_opfee,0)
                if original_volume[j]>0: # 先算有指定开仓数量的期权合约
                    set_deposit = np.append(set_deposit,td_deposit[-1])
                    other_deposit = np.append(other_deposit,0)
                    td_volume_cp = np.append(td_volume_cp,original_volume[j])
                    set_volume_pct = np.append(set_volume_pct,original_volume[j]/total_volume)
                    total_deposit = total_deposit+td_deposit[-1]*original_volume[j]
                elif original_volume[j]<0:
                    set_deposit = np.append(set_deposit,0)
                    other_deposit = np.append(other_deposit,td_deposit[-1])
                    td_volume_cp = np.append(td_volume_cp,0)
                    set_volume_pct = np.append(set_volume_pct,0)
                else:
                    set_deposit = np.append(set_deposit,0)
                    other_deposit = np.append(other_deposit,0)
                    td_volume_cp = np.append(td_volume_cp,-1)
                    set_volume_pct = np.append(set_volume_pct,0)
            if pct*Capital<=total_deposit: # 如果用来开仓的钱都不足以支付指定的开仓数量的话，则按比例减少
                minus_volume = ma.ceil((total_deposit-pct*Capital)/sum(set_deposit*set_volume_pct)) # 这里其实是有误差的，我们只能假设每张期权的保证金出入不大
                minus_volume = np.array([ma.ceil(minus_volume*k) for k in set_volume_pct])
                td_volume_cp = td_volume_cp-minus_volume
            else: # 若有剩余资金，就均等地开仓于剩下的合约仓位中
                if (len(other_deposit)!=0) and (sum(other_deposit)!=0):
                    max_volume = int((pct*Capital-total_deposit)/sum(other_deposit))
                    td_volume_cp = np.array([k+max_volume for k in td_volume_cp if k==0])
            td_volume_cp[td_volume_cp<0] = 0
            td_deposit = (td_deposit-total_fee)*td_volume_cp
            td_fee = sum(td_fee*td_volume_cp)
            td_opfee = sum(td_opfee*td_volume_cp)
            td_capital = capital[-1]-td_opfee-td_fee
            td_strategy = capital[-1]-td_fee
            td_volume = []
            for j in range(m):
                td_volume.append([td_volume_cp[j]])
            
        elif signalDf['signal'].iloc[i]==0: # 不操作
            td_log = '--'
            td_price = price[-1]
            td_volume = volume[-1]
            td_direction = direction[-1]
            td_deposit = deposit[-1]
            if flag:
                td_fee,td_cover_profit = {},{} # 用来记录强制平仓或减仓时出现的手续费与收益落袋
                td_profit,td_fee[0] = CalProfit(signalDf['price'].iloc[i],td_price,td_volume,td_direction,Fee,Point) # 对于策略的每日收益来讲，每天都是平仓日，每天都要缴纳手续费
                td_volume,td_fee[1],td_cover_profit[0] = CalDeposit(signalDf['price'].iloc[i],td_price,signalDf['deposit'].iloc[i],td_deposit,td_volume,td_direction,Fee,Point)
                td_price,td_deposit,td_volume,td_fee[2],td_cover_profit[1],td_opfee = CalPosition(signalDf['price'].iloc[i],td_price,signalDf['deposit'].iloc[i],td_deposit,\
                                                    td_volume,td_direction,signalDf['position'].iloc[i],signalDf['pct'].iloc[i],Fee,Point,capital[-1])
                td_capital = capital[-1]-td_fee[1]-td_fee[2]+sum(td_cover_profit)-td_opfee
                td_strategy = capital[-1]+opfee[-1]+td_profit-td_fee[0]-td_fee[2]
                td_fee = td_fee[1]+td_fee[2]
                td_opfee = opfee[-1]+td_opfee
            else:
                td_profit = 0
                td_fee = 0
                td_opfee = opfee[-1]
                td_capital = capital[-1]
                td_strategy = strategy[-1]
        
        else: # 平仓
            flag = False
            td_log = '所有仓位平仓'
            td_price = price[-1]
            td_volume = volume[-1]
            td_direction = direction[-1]
            td_profit,td_fee = CalProfit(signalDf['price'].iloc[i],td_price,td_volume,td_direction,Fee,Point)
            td_strategy = capital[-1]+opfee[-1]+td_profit-td_fee
            td_capital = td_strategy # 平仓时落袋为安
            td_price,td_volume,td_direction,td_deposit,td_opfee = [[0]],[[0]],np.array([0]),np.array([0]),0 # 清空记录
        
        trade_date.append(td)
        profit.append(td_profit)
        price.append(td_price)
        volume.append(td_volume)
        direction.append(td_direction)
        deposit.append(td_deposit)
        fee.append(td_fee)
        opfee.append(td_opfee)
        log.append(td_log)
        capital.append(td_capital)
        strategy.append(td_strategy)
        
        if capital[-1]<0:
            break
        
    return pd.DataFrame({'trade_date':trade_date,'signal':signal,'profit':profit,'price':price,'volume':volume,'direction':direction,\
                         'deposit':deposit,'fee':fee,'opfee':opfee,'log':log,'capital':capital,'strategy':strategy})

#%%

def CalProfit(td_price,open_price,volume,direction,Fee,Point):
    
    profit,fee = 0,0
    n = len(direction)
    for i in range(n):
        m = len(open_price[i]) # 同一张合约下面有可能交易了多个仓位（加仓）
        for j in range(m):
            profit = profit+direction[i]*(td_price[i]-open_price[i][j])*volume[i][j]*Point
            fee = fee+volume[i][j]*Fee
    
    return profit,fee


def CalDeposit(td_price,open_price,td_deposit,deposit,volume,direction,Fee,Point):
    
    volume_array = []
    fee,cover_profit = 0,0
    n = len(direction)
    for i in range(n):
        new_volume = co.deepcopy(volume[i])
        new_volume = np.array(new_volume)
        if direction[i]<0: # 只对空头仓进行保证金的计算
            new_deposit = 0
            m = len(new_volume)
            for j in range(m):
                new_deposit = new_deposit+td_deposit[i]*new_volume[j]
            if new_deposit>deposit[i]: # 本仓位的保证金已超过账户余额，必须强制平仓
                volume_pct = new_volume/sum(new_volume)
                minus_volume = ma.ceil((new_deposit-deposit[i])/td_deposit[i]) # 还是按比例来平仓
                minus_volume = np.array([ma.ceil(minus_volume*k) for k in volume_pct])
                new_volume = new_volume-minus_volume
                fee = fee+sum(minus_volume*Fee)
                new_price = np.array([])
                for j in range(m):
                    new_price = np.append(new_price,td_price[i]-open_price[i][j])
                cover_profit = cover_profit+sum(new_price*minus_volume*Point)
        volume_array.append(list(new_volume))
    
    return volume_array,fee,cover_profit


def CalPosition(td_price,price,td_deposit,deposit,volume,direction,position,pct,Fee,Point,capital):
    
    cover_profit,fee,opfee = 0,0,0
    volume_array = []
    n = len(direction)
    for i in range(n):
        if position[i]<0: # 减仓，先减仓会带来收益，可以为后续加仓提供更多可开的资金
            new_volume = co.deepcopy(volume[i])
            if type(pct[i])==float:
                cover_volume = sum(volume[i])*pct[i]
            else:
                cover_volume = pct[i]
            if cover_volume<=volume[i][0]:
                new_volume[0] = new_volume[0]-cover_volume
                cover_profit = cover_profit+direction[i]*(td_price[i]-price[i][0])*cover_volume*Point
                fee = fee+cover_volume*Fee
                if direction[i]>0: # 多头的减仓需要返回部分先前支付的期权费，加上今日收益才是平仓后的既得利益
                    opfee = opfee-price[i][0]*cover_volume*Point
            else:
                sum_volume = volume[i][0]
                m = len(volume[i])
                for j in range(1,m,1):
                    sum_volume = sum_volume+volume[i][j]
                    if cover_volume<=sum_volume:
                        for k in range(j-1):
                            cover_profit = cover_profit+direction[i]*(td_price[i]-price[i][k])*new_volume[k]*Point
                            if direction[i]>0:
                                opfee = opfee-price[i][k]*new_volume[k]*Point
                            new_volume[k] = 0
                        cover_profit = cover_profit+direction[i]*(td_price[i]-price[i][j])*(new_volume[j]-sum_volume+cover_volume)*Point
                        if direction[i]>0:
                            opfee = opfee-price[i][j]*(new_volume[j]-sum_volume+cover_volume)*Point
                        new_volume[j] = sum_volume-cover_volume
                        fee = fee+cover_volume*Fee
                        break
                    else:
                        if j<m-1:
                            continue
                        for k in range(j):
                            cover_profit = cover_profit+direction[i]*(td_price[i]-price[i][k])*new_volume[k]*Point
                            if direction[i]>0:
                                opfee = opfee-price[i][k]*new_volume[k]*Point
                            new_volume[k] = 0
                        fee = fee+sum(volume[i])*Fee
            volume_array.append(new_volume)
        else:
            volume_array.append(volume[i])
    new_capital = capital+cover_profit-fee-opfee
    
    add_volume = np.array([])
    add_deposit = np.array([])
    add_total_fee = np.array([])
    add_fee = np.array([])
    for i in range(n):
        if position[i]>0: # 加仓
            if type(pct[i])==float: # 如果给定小数点，则按第一个仓位的百分比开仓
                add_volume = np.append(add_volume,int(volume[i][0]*pct[i]))
            else:
                add_volume = np.append(add_volume,pct[i])
            if direction[i]>0: # 做多
                add_deposit = np.append(add_deposit,td_price[i]*Point+2*Fee)
                add_total_fee = np.append(add_fee,2*Fee)
                add_fee = np.append(add_fee,Fee)
            elif direction[i]>0: # 做空
                add_deposit = np.append(add_deposit,td_deposit[i]+Fee)
                add_total_fee = np.append(add_fee,Fee)
                add_fee = np.append(add_fee,0)
            else: # 若原本合约并没有开仓，但是加仓处有成交量，则按正负划分多空头
                if position[i]>=1:
                    add_deposit = np.append(add_deposit,td_price[i]*Point+2*Fee)
                    add_total_fee = np.append(add_fee,2*Fee)
                    add_fee = np.append(add_fee,Fee)
                elif position[i]<=-1:
                    add_deposit = np.append(add_deposit,td_deposit[i]+Fee)
                    add_total_fee = np.append(add_fee,Fee)
                    add_fee = np.append(add_fee,0)
                else:
                    add_deposit = np.append(add_deposit,0)
                    add_total_fee = np.append(add_fee,0)
                    add_fee = np.append(add_fee,0)
        else:
            add_volume = np.append(add_volume,0)
            add_deposit = np.append(add_deposit,0)
            add_total_fee = np.append(add_fee,0)
            add_fee = np.append(add_fee,0)
    if sum(add_deposit*add_volume)>new_capital:
        minus_volume = ma.ceil((sum(add_deposit*add_volume)-new_capital)/sum(add_deposit*(add_volume/sum(add_volume))))
        minus_volume = np.array([ma.ceil(minus_volume*k) for k in (add_volume/sum(add_volume))])
        add_volume = add_volume-minus_volume
    deposit_array = np.array([])
    price_array = co.deepcopy(price)
    for i in range(n):
        if position[i]>0:
            deposit_array = np.append(deposit_array,deposit[i]+(add_deposit[i]-add_total_fee[i])*add_volume[i])
            volume_array[i].append(add_volume[i])
            price_array[i].append(td_price[i])
            if direction[i]>0:
                opfee = opfee+td_price[i]*add_volume[i]*Point
        else:
            deposit_array = np.append(deposit_array,deposit[i])
    fee = fee+sum(add_fee)
    
    return price_array,deposit_array,volume_array,fee,cover_profit,opfee

#%%

def Visualize(recordDf,benchmarkDf,title='回测结果'):
    
    strategy = co.deepcopy(recordDf[['trade_date','signal','capital','strategy']])
    benchmark = co.deepcopy(benchmarkDf)
    
    capital = strategy['capital'].iloc[0]
    S0 = benchmark['close'].iloc[0]
    
    strategy['pct_strategy'] = 100*(strategy['strategy']-capital)/capital
    benchmark['pct_benchmark'] = 100*(benchmark['close']-S0)/S0
    
    plt.figure(figsize=(12,4))
    plt.plot(strategy['trade_date'],strategy['pct_strategy'],label='策略收益%')
    plt.plot(benchmark['trade_date'],benchmark['pct_benchmark'],label='基准线收益%')
    plt.legend(loc='upper left')
    plt.title(title)
    
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