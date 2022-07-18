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

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def calculate_position(raw_data, *args):
    rf = args[0]
    k = args[1]
    data = raw_data[['交易日期', '收盘价', '前收盘价']].rename(columns={'交易日期': 'date',
                                                             '收盘价': 'closePrice',
                                                             '前收盘价': 'lastClosePrice'})
    data['date'] = data.date.apply(np.datetime64)
    data = data.set_index('date')
    data['ret'] = np.log(data.closePrice).diff(1)
    data['long_signal'] = data.ret.rolling('7d').apply(lambda s: s[-1]-np.var(s)/2 > rf)*1
    data['out_signal'] = (np.exp(data.ret) > (1+k)*np.exp(rf/252))*1
    data['signal'] = np.maximum(data.long_signal-data.out_signal, 0)
    data['position'] = data.signal.shift(1)
    return data


def calculate_pnl(data, cost):
    day_pnl = data.position*(data.closePrice-data.lastClosePrice)
    trans_cost = (data.signal != data.position)*cost
    data['day_pnl'] = day_pnl
    data['trans_cost'] = trans_cost
    data['pnl_with_trans'] = data['day_pnl']-data['trans_cost']
    return data


# backtest
path = os.getcwd()+'/stock_data'
files = os.listdir(path)

rf = np.log(1+0.02)
k = 1
transaction_cost = 0.0006

pnl_with_trans = {}
for file in files:
    file_name = file.split('.')[0]
    raw_data = pd.read_csv(path+'/'+file, skiprows=1, encoding='GBK')
    data = calculate_position(raw_data, rf, k)
    port = calculate_pnl(data, transaction_cost)
    pnl_accumulated = port.copy().pnl_with_trans.fillna(0)

    for i in range(1, len(pnl_accumulated)):
        reinvest_coef = np.exp((pnl_accumulated.index[i]-pnl_accumulated.index[i-1]).days/252*rf)
        last_pnl_accumulated = pnl_accumulated[i-1]*reinvest_coef
        pnl_accumulated[i] = last_pnl_accumulated + pnl_accumulated[i]

    # contract
    pnl_holding_stock = port.closePrice-port.closePrice[0]*np.exp((port.index-port.index[0]).days/252*rf)
    pnl_with_trans[file_name] = pd.Series(pnl_accumulated, index=port.index.values)

    figure = plt.figure(figsize=(8, 6))
    plt.plot(port.index.values, pnl_accumulated, label='strategy pnl')
    plt.plot(port.index.values, pnl_holding_stock.values, label='stock holding pnl')
    plt.xlabel('date')
    plt.ylabel('profit/Chinese yen')
    plt.xticks(rotation=45)
    plt.title(file_name+' accumulated pnl')
    plt.legend()
    plt.savefig(file_name+'.png')

pnls_with_trans = pd.DataFrame(pnl_with_trans).fillna(0)
pnl_accumulated = pnls_with_trans.copy()

