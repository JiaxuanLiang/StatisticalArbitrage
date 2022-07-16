# -*- coding: utf-8 -*- 
"""
--------------------------------------------------
File Name:        MonteCarloShort
Description:    
Author:           jiaxuanliang
Date:             7/16/22
--------------------------------------------------
Change Activity:  7/16/22
--------------------------------------------------
"""
import matplotlib.pyplot as plt
import numpy as np
from Strategies.ShortStrategy import Short

seed = 42

# short until deterministic boundary example
# Monte Carlo
# params
alpha = .01
rf = .05
sig = .2
s0 = 1
k = 0.05
n_sim = 10000
T = np.array([1, 2, 5, 10, 20, 50])
n_step = 252*T[-1]
dt = 1/252

# set strategy
strategy_short = Short(rf, s0, alpha, sig, seed)
s_path = strategy_short.simulate_path(shape=(n_sim, n_step), dt=dt, update=False)


# given timestamps, get discounted payoff
bucket = []
mean = []
time_averaged_var = []
prob_of_loss = []

for t in T:
    vt = strategy_short.discount_payoff(t, barrier_type='deterministic', bound=k)
    bucket.append(vt)
    statistics = strategy_short.statistics_of_simulated_discounted_payoff(t)
    mean.append(statistics.get('mean'))
    time_averaged_var.append(statistics.get('time averaged var'))
    prob_of_loss.append(statistics.get('probability of loss'))

# plot figure8
figure8, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey='all', figsize=(12, 8))
ax1.hist(bucket[0], bins=12)
ax1.set_title('One Year', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax2.hist(bucket[1], bins=12)
ax2.set_title('Two Year', fontsize=12)
ax3.hist(bucket[2], bins=12)
ax3.set_title('Five Year', fontsize=12)
ax4.hist(bucket[3])
ax4.set_title('Ten Year', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_xlabel('Profit', fontsize=12)
ax5.hist(bucket[4])
ax5.set_title('Twenty Year', fontsize=12)
ax5.set_xlabel('Profit', fontsize=12)
ax6.hist(bucket[5])
ax6.set_title('Fifty Year', fontsize=12)
ax6.set_xlabel('Profit', fontsize=12)
plt.savefig('Figure8.png')
plt.show()

# plot figure9
figure9, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
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
ax3.set_title('Probability of Loss', fontsize=12)
ax3.set_xlabel('Time(Years)', fontsize=12)
ax3.legend()
plt.savefig('Figure9.png')
plt.show()
