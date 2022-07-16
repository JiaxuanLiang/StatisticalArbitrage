# -*- coding: utf-8 -*- 
"""
--------------------------------------------------
File Name:        ShortStrategy
Description:    
Author:           jiaxuanliang
Date:             7/16/22
--------------------------------------------------
Change Activity:  7/16/22
--------------------------------------------------
"""
import numpy as np
from BlackScholesSetting.Models import StockModels


class Short:
    def __init__(self, rf, s0, alpha, sigma, seed):
        """
        general buy and hold strategy
        :param rf: money market rate
        :param alpha: stock return
        :param sigma: stock return vol (standard deviation)
        :param seed: to produce stable result
        """
        if alpha - rf < sigma**2/2:
            self.rf = rf
            self.stock = StockModels.StockGBM(s0=s0, alpha=alpha, sig=sigma, seed=seed)
            self.discount_stock = StockModels.StockGBM(s0=s0, alpha=alpha-rf, sig=sigma, seed=seed)
            self.tau = None
            self.discounted_profit = None
        else:
            raise ValueError('alpha should be relatively smaller for short strategy')

    def simulate_path(self, shape, dt=1/252, update=False):
        path = self.stock.simulate_path(shape, dt, update)
        _ = self.discount_stock.simulate_path(shape, dt, update)
        return path

    def discount_payoff(self, t, barrier_type, bound=np.inf):
        """
        discounted payoff of strategy
        :param t: terminate time t
        :param barrier_type: "none" , "deterministic"
        :param bound: parameter of the boundary condition on price,
                      for "deterministic", the boundary is S0/(1+bound) for discounted_St
        :return: discounted_payoff of portfolio should have the length of n_sim
        """
        terminate = t/self.stock.dt
        shape = self.stock.path.shape
        self.tau = t
        if barrier_type == 'none':
            if len(shape) == 1:
                self.discounted_profit = -(self.discount_stock.path[terminate]-self.stock.s0)
            else:
                self.discounted_profit = -(self.discount_stock.path[:, terminate]-self.stock.s0)

        if barrier_type == 'deterministic':
            new_bound = self.stock.s0/(1+bound)
            t_in, index_in = self.discount_stock.max_time_and_index_in_boundary(new_bound, bound_type='down')
            if len(shape) == 1:
                if index_in < terminate:
                    self.discounted_profit = bound*self.stock.s0/(1+bound)
                else:
                    self.discounted_profit = -self.discount_stock.path[int(terminate)]+self.discount_stock.s0
            else:
                vt = []
                for dis_path, i in zip(self.discount_stock.path, index_in):
                    if i < terminate:
                        vt.append(bound*self.stock.s0/(1+bound))
                    else:
                        vt.append(-dis_path[int(terminate)]+self.stock.s0)
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
                      meaning the bound will be considered as s0exp(rt)/(1+bound)
        :return: dict, with keys-'expectation', 'expectation limit',
                 'variance', 'variance limit',
                 'time averaged variance', 'time averaged variance limit',
                 'probability of loss', 'probability of loss limit'
        """
        no_hit_expected_v = -(np.exp(self.discount_stock.alpha*t)-1)*self.discount_stock.s0
        item1 = np.exp(self.discount_stock.sig ** 2) - 1
        item2 = self.discount_stock.s0 ** 2
        item3 = np.exp(2 * self.discount_stock.alpha * t)
        no_hit_var_v = item1 * item2 * item3
        no_hit_prob_of_loss = 1-self.discount_stock.value_cdf(t, self.discount_stock.s0)

        if barrier_type == 'none':
            time_averaged_var = no_hit_var_v/t
            if self.discount_stock.alpha < 0:
                lim_expectation = self.stock.s0
                if self.discount_stock.alpha <= -self.discount_stock.sig**2/2:
                    lim_var_v = 0
                    lim_time_averaged_var = 0
                else:
                    lim_var_v = None
                    lim_time_averaged_var = None
            else:
                lim_expectation = None

            statistics = {'expectation': no_hit_expected_v,
                          'expectation limit': lim_expectation,
                          'variance': no_hit_var_v,
                          'variance limit': lim_var_v,
                          'time averaged var': time_averaged_var,
                          'time averaged var limit': lim_time_averaged_var,
                          'probability of loss': no_hit_prob_of_loss}
            return statistics

        if barrier_type == 'deterministic':
            new_bound = self.stock.s0/(1+bound)

            hit_expected_v = self.stock.s0*bound/(1+bound)
            hit_prob = self.discount_stock.analytical_first_exceeding_time_cdf(new_bound, t)
            not_hit_prob = 1 - hit_prob

            expected_v = hit_expected_v*hit_prob+not_hit_prob*no_hit_expected_v
            var_v = not_hit_prob*no_hit_var_v
            time_averaged_var = var_v/t
            prob_of_loss = not_hit_prob*no_hit_prob_of_loss

            if bound > 0:
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
                raise ValueError('bound should be positive')
