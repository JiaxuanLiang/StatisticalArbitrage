# -*- coding: utf-8 -*- 
"""
--------------------------------------------------
File Name:        MonteCarloBuyHoldConstantBarrierDemo
Description:      relationship between v(t) for different alphas and sigmas
Author:           jiaxuanliang
Date:             7/16/22
--------------------------------------------------
Change Activity:  7/16/22
--------------------------------------------------
"""
import matplotlib.pyplot as plt
from Strategies.BuyHoldStrategy import *

seed = 42


# buy and hold until constant barrier example
# plot figure1
s0_test = 1
B_test = 1.3
rf_test = 0.05
alpha_test = np.linspace(0.13, 0.35, 21)
sig_test = np.linspace(0.1, 0.5, 5)
sig, alpha = np.meshgrid(sig_test, alpha_test)
shape = sig.shape
sig_f = sig.flatten()
alpha_f = alpha.flatten()


def get_expected_v(sig, alpha):
    ev = []
    for s, a in zip(sig, alpha):
        strategy_test = BuyHold(rf_test, s0_test, a, s, seed)
        v = strategy_test.statistics_of_analytical_discounted_profit(barrier_type='constant', bound=B_test, t=None)
        ev.append(v.get('expectation limit'))
    return np.array(ev).reshape(shape)


e_v = get_expected_v(sig_f, alpha_f)

figure1 = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(alpha, sig, e_v, cmap='jet')
ax.set_title('Expected Trading Profits')
ax.set_xlabel('$\\sigma$')
ax.set_ylabel('$\\alpha$')
ax.set_zlabel('Profit')
ax.view_init(23, 240)
plt.savefig('Figure1.png')
plt.show()
