# -*- coding: utf-8 -*- 
"""
--------------------------------------------------
File Name:        MonteCarloBuyHoldStatisticalArbitrage
Description:    
Author:           jiaxuanliang
Date:             7/14/22
--------------------------------------------------
Change Activity:  7/14/22
--------------------------------------------------
"""

import matplotlib.pyplot as plt
from Strategies.BuyHoldStrategy import *

seed = 42


# buy and hold until deterministic boundary example
# statistical arbitrage exists when
alpha = .05  # changed
rf = .04
sig = .2
s0 = 1
k = 0.05
n_sim = 10000
T = np.array([1, 2, 5, 10, 20, 50])
n_step = 252*T[-1]
dt = 1/252

# set strategy
strategy_buy_hold = BuyHold(rf, s0, alpha, sig, seed)
s_path = strategy_buy_hold.simulate_path(shape=(n_sim, n_step), dt=dt, update=False)

# given timestamps, get discounted payoff
bucket = []
mean = []
time_averaged_var = []
anal_time_averaged_var = []
prob_of_loss = []
anal_prob_of_loss = []
anal_prob_of_loss_limit = []

for t in T:
    vt = strategy_buy_hold.discount_payoff(t, barrier_type='deterministic', bound=k)
    bucket.append(vt)
    statistics = strategy_buy_hold.statistics_of_simulated_discounted_payoff(t)
    anal_stats = strategy_buy_hold.statistics_of_analytical_discounted_profit(barrier_type='deterministic',
                                                                              bound=k,
                                                                              t=t)
    mean.append(statistics.get('mean'))
    time_averaged_var.append(statistics.get('time averaged var'))
    prob_of_loss.append(statistics.get('probability of loss'))
    anal_prob_of_loss.append(anal_stats.get('probability of loss'))
    anal_prob_of_loss_limit.append(anal_stats.get('probability of loss limit'))

# plot figure10
figure10, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
ax1.plot(T, mean, label='Monty Carlo simulation')
ax1.set_title('Mean of Trading Profits', fontsize=12)
ax1.set_xlabel('Time(Years)', fontsize=12)
ax1.set_ylabel('Profit', fontsize=12)
ax1.legend()
ax2.plot(T, time_averaged_var, label='Monty Carlo simulation')
ax2.legend()
ax2.set_title('Time Averaged Variance', fontsize=12)
ax2.set_xlabel('Time(Years)', fontsize=12)
ax3.plot(T, prob_of_loss, label='Monty Carlo simulation')
ax3.plot(T, anal_prob_of_loss, '--', label='Analytical formula')
ax3.plot(T, anal_prob_of_loss_limit, '--', label='Limiting prob. of loss')
ax3.set_title('Probability of Loss', fontsize=12)
ax3.set_xlabel('Time(Years)', fontsize=12)
ax3.legend()
plt.savefig('Figure10.png')
plt.show()
