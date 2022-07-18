# -*- coding: utf-8 -*- 
"""
--------------------------------------------------
File Name:        strategy_backtest
Description:    
Author:           jiaxuanliang
Date:             7/17/22
--------------------------------------------------
Change Activity:  7/17/22
--------------------------------------------------
"""

# import packages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# calculate position based on signals
# here the position equals to the signal of the previous day
# suppose position is completed with the close price of previous day
# and the profit & loss is calculated in the day
def calculate_position(raw_data, *args):
    rf = args[0]
    k = args[1]
    data = raw_data[['交易日期', '收盘价', '前收盘价']].rename(columns={'交易日期': 'date',
                                                             '收盘价': 'closePrice',
                                                             '前收盘价': 'lastClosePrice'})
    data['date'] = data.date.apply(np.datetime64)
    data = data.set_index('date')
    data['ret'] = np.log(data.closePrice/data.lastClosePrice)
    # the condition for buy hold strategy
    data['long_signal'] = data.ret.rolling('7d').apply(lambda s: s[-1]-np.var(s)/2 > rf)*1
    data['out_signal'] = (np.exp(data.ret) > (1+k)*np.exp(rf/252))*1
    data['signal'] = np.maximum(data.long_signal-data.out_signal, 0)
    data['position'] = data.signal.shift(1).fillna(0)
    return data


# calculate pnl for each stock
def calculate_pnl(data, cost):
    day_pnl = data.position*(data.closePrice-data.lastClosePrice)
    # if today's signal indicate a different position for tomorrow different from today
    # transaction happens at the close price of today
    trans_cost = (data.signal != data.position)*cost
    data['day_pnl'] = day_pnl
    data['trans_cost'] = trans_cost
    data['pnl_with_trans'] = data['day_pnl']-data['trans_cost']*data['closePrice']
    return data


# backtest
path = os.getcwd()+'/stock_data'
files = os.listdir(path)

rf = np.log(1+0.02)
k = 1
transaction_cost = 0.0006

for file in files:
    file_name = file.split('.')[0]  # the stock id
    raw_data = pd.read_csv(path+'/'+file, skiprows=1, encoding='GBK')
    data = calculate_position(raw_data, rf, k)
    port = calculate_pnl(data, transaction_cost)
    # data is cleaned so nan will happen only when there is no transaction
    pnl_accumulated = port.copy().pnl_with_trans.fillna(0)

    for i in range(1, len(pnl_accumulated)):
        reinvest_coef = np.exp((pnl_accumulated.index[i]-pnl_accumulated.index[i-1]).days/252*rf)
        last_pnl_accumulated = pnl_accumulated[i-1]*reinvest_coef
        pnl_accumulated[i] = last_pnl_accumulated + pnl_accumulated[i]

    trans_dates = port[port.trans_cost != 0].index
    in_date = trans_dates[::2]
    in_accumu_pnl = pnl_accumulated[in_date]
    out_date = trans_dates[1::2]
    out_accumu_pnl = pnl_accumulated[out_date]

    # by contract, keep holding the stock at the first day financed by the money account
    pnl_holding_stock = port.closePrice-port.closePrice[0]*np.exp((port.index-port.index[0]).days/252*rf)

    # make plots
    figure = plt.figure(figsize=(20, 8))
    plt.plot(port.index.values, pnl_accumulated, label='strategy pnl')
    plt.plot(port.index.values, pnl_holding_stock.values, label='stock holding pnl')
    plt.scatter(in_date.values, in_accumu_pnl.values, c='green', marker='.', alpha=0.4, label='long point')
    plt.scatter(out_date.values, out_accumu_pnl.values, c='red', marker='.', alpha=0.4, label='out point')
    plt.xlabel('date')
    plt.ylabel('profit/Chinese yen')
    plt.xticks(rotation=30)
    plt.title(file_name+' accumulated pnl')
    plt.legend()
    plt.savefig(file_name+'.png')
