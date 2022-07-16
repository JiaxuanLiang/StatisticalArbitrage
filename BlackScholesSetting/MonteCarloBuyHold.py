# -*- coding: utf-8 -*- 
"""
--------------------------------------------------
File Name:        Monte Carlo
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

# buy and hold until constant barrier example
# Monty Carlo
# params
alpha = .16
rf = .04
sig = .2
s0 = 1
B = 1.2
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
prob_of_loss = []

for t in T:
    vt = strategy_buy_hold.discount_payoff(t, barrier_type='constant', bound=B)
    bucket.append(vt)
    statistics = strategy_buy_hold.statistics_of_simulated_discounted_payoff(t)
    anal_stats = strategy_buy_hold.statistics_of_analytical_discounted_profit(barrier_type='constant', bound=B, t=t)
    mean.append(statistics.get('mean'))
    time_averaged_var.append(statistics.get('time averaged var'))
    prob_of_loss.append(statistics.get('probability of loss'))

# plot figure2
figure2, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey='all', figsize=(12, 8))
ax1.hist(bucket[0], bins=12)
ax1.set_title('One Year', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax2.hist(bucket[1], bins=12)
ax2.set_title('Two Year', fontsize=12)
ax3.hist(bucket[2], bins=12)
ax3.set_title('Five Year', fontsize=12)
ax4.hist(bucket[3], bins=12)
ax4.set_title('Ten Year', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_xlabel('Profit', fontsize=12)
ax5.hist(bucket[4], bins=12)
ax5.set_title('Twenty Year', fontsize=12)
ax5.set_xlabel('Profit', fontsize=12)
ax6.hist(bucket[5], bins=12)
ax6.set_title('Fifty Year', fontsize=12)
ax6.set_xlabel('Profit', fontsize=12)
plt.savefig('Figure2.png')
plt.show()

# plot figure3
figure3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
ax1.plot(T, mean, label='Monty Carlo simulation')
ax1.hlines(anal_stats.get('expectation'), T[0], T[-1], linestyles='dashed', label='Analytical limit')
ax1.set_title('Mean of Trading Profits', fontsize=12)
ax1.set_xlabel('Time(Years)', fontsize=12)
ax1.set_ylabel('Profit', fontsize=12)
ax1.legend()
ax2.plot(T, time_averaged_var, label='Monty Carlo simulation')
ax2.plot(T, anal_stats.get('variance limit')/T, '--', label='Analytical limit')
ax2.legend()
ax2.set_title('Time Averaged Variance', fontsize=12)
ax2.set_xlabel('Time(Years)', fontsize=12)
ax3.plot(T, prob_of_loss, label='Monty Carlo simulation')
ax3.hlines(anal_stats.get('probability of loss'), T[0], T[-1], linestyles='dashed', label='Analytical limit')
ax3.set_title('Probability of Loss', fontsize=12)
ax3.set_xlabel('Time(Years)', fontsize=12)
ax3.legend()
plt.savefig('Figure3.png')
plt.show()

# plot figure4
tau = np.linspace(0.001, 10, int(10/dt))
density = strategy_buy_hold.stock.analytical_first_exceeding_time_pdf(B, tau)
figure4 = plt.figure(figsize=(10, 4))
ax = figure4.gca()
ax.plot(tau, density)
ax.set_title('Inverse Gaussian PDF (Time)', fontsize=12)
ax.set_xlabel('Time', fontsize=12)
plt.savefig('Figure4.png')
plt.show()


# buy and hold until deterministic boundary example
# Monte Carlo
# params
alpha = .16
rf = .04
sig = .2
s0 = 1
k = 0.05
n_sim = 10000
T = np.array([1, 2, 5, 10, 20, 50])
n_step = 252*T[-1]
dt = 1/252

# plot figure 5
path_example = s_path[:10, :int(1/dt)+1]
t = np.linspace(0, 1, int(1/dt)+1)
sell_bound = s0*(1+k)*np.exp(rf*t)
rf_grow = s0*np.exp(rf*t)
expected_price = s0*np.exp(alpha*t)
figure5 = plt.figure(figsize=(16, 8))
for path in path_example:
    plt.plot(t, path)
plt.plot(t, sell_bound, '--', label='sell if it hits')
plt.plot(t, rf_grow, '--', label='risk free growth')
plt.plot(t, expected_price, label='expected growth of the stock')
plt.xlabel('Time in years')
plt.ylabel('Stock price')
plt.legend()
plt.savefig('Figure5.png')
plt.show()

# given timestamps, get discounted payoff
bucket = []
mean = []
time_averaged_var = []
prob_of_loss = []

for t in T:
    vt = strategy_buy_hold.discount_payoff(t, barrier_type='deterministic', bound=k)
    bucket.append(vt)
    statistics = strategy_buy_hold.statistics_of_simulated_discounted_payoff(t)
    mean.append(statistics.get('mean'))
    time_averaged_var.append(statistics.get('time averaged var'))
    prob_of_loss.append(statistics.get('probability of loss'))

# plot figure6
figure6, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey='all', figsize=(12, 8))
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
plt.savefig('Figure6.png')
plt.show()

# plot figure7
figure7, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
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
plt.savefig('Figure7.png')
plt.show()
