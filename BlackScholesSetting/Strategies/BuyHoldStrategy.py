# -*- coding: utf-8 -*- 
"""
--------------------------------------------------
File Name:        strategies
Description:    
Author:           jiaxuanliang
Date:             7/10/22
--------------------------------------------------
Change Activity:  7/16/22
--------------------------------------------------
"""

import numpy as np
from BSsetting.Models import StockModels


class BuyHold:
    def __init__(self, rf, s0, alpha, sigma, seed):
        """
        general buy and hold strategy
        :param rf: money market rate
        :param alpha: stock return
        :param sigma: stock return vol (standard deviation)
        :param seed: to produce stable result
        """
        if alpha > rf:
            self.rf = rf
            self.stock = StockModels.StockGBM(s0=s0, alpha=alpha, sig=sigma, seed=seed)
            self.discount_stock = StockModels.StockGBM(s0=s0, alpha=alpha-rf, sig=sigma, seed=seed)
            self.tau = None
            self.discounted_profit = None
        else:
            raise ValueError('alpha should be greater than r for long strategy')

    def simulate_path(self, shape, dt=1/252, update=False):
        path = self.stock.simulate_path(shape, dt, update)
        _ = self.discount_stock.simulate_path(shape, dt, update)
        return path

    def discount_payoff(self, t, barrier_type, bound=np.inf):
        """
        discounted payoff of strategy
        :param t: terminate time t
        :param barrier_type: "none" , "constant", "deterministic"
        :param bound: parameter of the boundary condition on price,
                      float b for "constant", the boundary is b for St
                      float k for "deterministic", the boundary is S0(1+k) for discounted_St
        :return: discounted_payoff of portfolio should have the length of n_sim
        """
        terminate = t/self.stock.dt
        shape = self.stock.path.shape
        self.tau = t
        if barrier_type == 'none':
            if len(shape) == 1:
                self.discounted_profit = self.discount_stock.path[terminate]-self.stock.s0
            else:
                self.discounted_profit = self.discount_stock.path[:, terminate]-self.stock.s0
        if barrier_type == 'constant':
            t_in, index_in = self.stock.max_time_and_index_in_boundary(bound, bound_type='up')
            if len(shape) == 1:
                if index_in < terminate:
                    self.discounted_profit = bound * np.exp(-self.rf*self.stock.dt*(index_in+1))-self.stock.s0
                else:
                    self.discounted_profit = self.discount_stock.path[int(terminate)] - self.stock.s0
            else:
                vt = []
                for discounted_path, i in zip(self.discount_stock.path, index_in):
                    if i < terminate:
                        vt.append(bound * np.exp(-self.rf * self.stock.dt * (i + 1)) - self.stock.s0)
                    else:
                        vt.append(discounted_path[int(terminate)] - self.stock.s0)
                self.discounted_profit = np.array(vt)
        if barrier_type == 'deterministic':
            new_bound = (1+bound)*self.stock.s0
            t_in, index_in = self.discount_stock.max_time_and_index_in_boundary(new_bound, bound_type='up')
            if len(shape) == 1:
                if index_in < terminate:
                    self.discounted_profit = bound*self.stock.s0
                else:
                    self.discounted_profit = self.discount_stock.path[int(terminate)]-self.discount_stock.s0
            else:
                vt = []
                for dis_path, i in zip(self.discount_stock.path, index_in):
                    if i < terminate:
                        vt.append(bound*self.stock.s0)
                    else:
                        vt.append(dis_path[int(terminate)]-self.stock.s0)
                self.discounted_profit = np.array(vt)
        return self.discounted_profit

    def statistics_of_simulated_discounted_payoff(self, t):
        """
        call self.discounted_payoff(self, t, barrier_type, bound) first
        """
        if isinstance(self.discounted_profit, np.ndarray):
            mean = np.mean(self.discounted_profit)
            var = np.var(self.discounted_profit)
            time_averaged_var = var/t
            prob_of_loss = np.sum(self.discounted_profit < 0) / len(self.discounted_profit)
            return {'mean': mean,
                    'var': var,
                    'time averaged var': time_averaged_var,
                    'probability of loss': prob_of_loss}
        else:
            print('does not calculate statistics for one value')

    def statistics_of_analytical_discounted_profit(self, barrier_type, t, bound):
        """
        :param barrier_type: 'none', 'constant', or 'deterministic'
        :param t: terminate point t
        :param bound: if 'none', bound is None
                      if 'constant', bound is the value of bound
                      if 'deterministic', bound is the percentage,
                      meaning the bound will be considered as s0(1+bound)exp(rt)
        :return: dict, with keys-'expectation', 'expectation limit',
                 'variance', 'variance limit',
                 'time averaged variance', 'time averaged variance limit',
                 'probability of loss', 'probability of loss limit'
        """
        no_hit_expected_v = (np.exp(self.discount_stock.alpha*t)-1)*self.discount_stock.s0
        item1 = np.exp(self.discount_stock.sig ** 2) - 1
        item2 = self.discount_stock.s0 ** 2
        item3 = np.exp(2 * self.discount_stock.alpha * t)
        no_hit_var_v = item1 * item2 * item3
        no_hit_prob_of_loss = self.discount_stock.value_cdf(t, self.discount_stock.s0)

        if barrier_type == 'none':
            time_averaged_var = no_hit_var_v/t
            statistics = {'expectation': no_hit_expected_v,
                          'variance': no_hit_var_v,
                          'time averaged var': time_averaged_var,
                          'probability of loss': no_hit_prob_of_loss}
            return statistics

        if barrier_type == 'constant':
            mu = self.stock.bm.mu
            s0 = self.stock.s0
            sig = self.stock.sig

            hit_expected_v = bound*(bound/s0)**((mu-np.sqrt(2*self.rf+mu**2))/sig)-s0
            item1 = (bound/s0)**((mu-np.sqrt(4*self.rf+mu**2))/sig)
            item2 = (bound/s0)**((2*mu-2*np.sqrt(2*self.rf+mu**2))/sig)
            hit_var_v = (bound**2)*(item1-item2)

            hit_prob = self.stock.analytical_first_exceeding_time_cdf(bound, t)
            not_hit_prob = 1-hit_prob

            expected_v = hit_expected_v*hit_prob+no_hit_expected_v*not_hit_prob
            var_v = hit_var_v*hit_prob+no_hit_var_v*not_hit_prob
            time_averaged_var = var_v/t

            hit_win_prob = self.stock.bm.analytical_first_exceeding_time_cdf(np.log(bound / s0) / sig,
                                                                             np.log(bound / s0) / self.rf)
            hit_prob_of_loss = hit_prob-hit_win_prob
            prob_of_loss = not_hit_prob*no_hit_prob_of_loss+(t >= np.log(bound/s0)/self.rf)*hit_prob_of_loss

            if s0 < bound and mu > 0:
                lim_prob_of_loss = 1-hit_prob_of_loss
                statistics = {'expectation': expected_v,
                              'expectation limit': hit_expected_v,
                              'variance': var_v,
                              'variance limit': hit_var_v,
                              'time averaged variance': time_averaged_var,
                              'time averaged variance limit': 0,
                              'probability of loss': prob_of_loss,
                              'probability of loss limit': lim_prob_of_loss}
                return statistics

            if s0 < bound and mu < 0:
                b_star = np.log(bound/s0)/self.stock.sig
                lim_probability_of_hit = np.exp(2*mu*b_star)
                lim_probability_of_not_hit = 1-lim_probability_of_hit
                lim_expected_v = lim_probability_of_hit*hit_expected_v+lim_probability_of_not_hit*(-self.stock.s0)
                lim_prob_of_loss = (lim_probability_of_hit-hit_win_prob)+(1-lim_probability_of_hit)
                statistics = {'expectation': expected_v,
                              'expectation limit': lim_expected_v,
                              'variance': var_v,
                              'variance limit': None,
                              'time averaged variance': time_averaged_var,
                              'time averaged variance limit': None,
                              'probability of loss': prob_of_loss,
                              'probability of loss limit': lim_prob_of_loss}
                return statistics

            else:
                raise ValueError('parameters do not apply')

        if barrier_type == 'deterministic':
            new_bound = self.stock.s0*(1+bound)
            hit_expected_v = self.stock.s0*bound
            hit_prob = self.discount_stock.analytical_first_exceeding_time_cdf(new_bound, t)
            not_hit_prob = 1 - hit_prob

            expected_v = hit_expected_v*hit_prob+not_hit_prob*no_hit_expected_v
            var_v = not_hit_prob*no_hit_var_v
            time_averaged_var = var_v/t
            prob_of_loss = not_hit_prob*no_hit_prob_of_loss

            if self.discount_stock.bm.mu > 0 and bound > 0:
                statistics = {'expectation': expected_v,
                              'expectation limit': hit_expected_v,
                              'variance': var_v,
                              'variance limit': 0,
                              'time averaged variance': time_averaged_var,
                              'time averaged variance limit': 0,
                              'probability of loss': prob_of_loss,
                              'probability of loss limit': 0}
                return statistics
            else:
                k_star = np.log(1+bound)/self.discount_stock.sig
                lim_prob_of_loss = 1-np.exp(2*k_star*self.discount_stock.bm.mu)
                lim_expected_v = -lim_prob_of_loss*self.stock.s0+(1-lim_prob_of_loss)*bound*self.stock.s0
                statistics = {'expectation': expected_v,
                              'expectation limit': lim_expected_v,
                              'variance': var_v,
                              'variance limit': None,
                              'time averaged variance': time_averaged_var,
                              'time averaged variance limit': None,
                              'probability of loss': prob_of_loss,
                              'probability of loss limit': lim_prob_of_loss}
                return statistics

    def analytical_discounted_profit_to_bound_derivative_constant_barrier(self, bound):
        if (s0 := self.stock.s0) < bound:
            sig = self.stock.sig
            alpha = self.stock.alpha
            mu = (alpha-sig**2/2)/sig
            return (sig+mu-np.sqrt(2*self.rf+mu**2))/sig*(bound/s0)**((mu-np.sqrt(self.rf+mu**2))/sig)
        else:
            raise ValueError('bound should be bigger than s0')


if __name__ == '__main__':
    seed = 42
    alpha = .16
    rf = .04
    sig = .2
    s0 = 1
    B = 1.2
    n_sim = 10
    T = np.array([1, 2])
    n_step = 252 * T[-1]
    dt = 1 / 252

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
        mean.append(statistics.get('mean'))
        time_averaged_var.append(statistics.get('time averaged var'))
        prob_of_loss.append(statistics.get('probability of loss'))
