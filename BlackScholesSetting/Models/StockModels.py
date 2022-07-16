# -*- coding: utf-8 -*- 
"""
--------------------------------------------------
File Name:        entity
Description:    
Author:           jiaxuanliang
Date:             7/10/22
--------------------------------------------------
Change Activity:  7/10/22
--------------------------------------------------
"""

import numpy as np
from BrownianMotion import BM


# stocks
class StockGBM:
    def __init__(self, s0, alpha, sig, seed):
        if sig > 0:
            self.s0 = s0
            self.alpha = alpha
            self.sig = sig
            self.bm = BM(drift=(alpha-sig**2/2)/sig, seed=seed)
            self.path = None
            self.dt = None
            self.last_simulated_path = None
            self.last_simulated_dt = None
        else:
            raise ValueError('sigma should be positive')

    # stock price path GBM simulation
    def simulate_path(self, shape, dt, update):
        """
        monte carlo simulation
        :param shape: int, or tuple in the form of (n_simulation, n_time_step)
        :param dt: step length, time interval, default daily (1/252)
        :param update: bool, whether to save the simulated path, default False.
                       To have stable result, turn to True and access through object.read_path()
        :return: paths
        """
        log_s = self.bm.simulate_path(shape=shape, dt=dt, update=update)*self.sig
        self.last_simulated_path = self.s0*np.exp(log_s)
        self.last_simulated_dt = dt
        if any([update, self.path is None]):
            self.dt = self.last_simulated_dt
            self.path = self.last_simulated_path.copy()
        return self.last_simulated_path

    def max_time_and_index_in_boundary(self, bound, bound_type):
        """
        largest time_and index in boundary of the saved path
        :param bound: int, float or tuple
        :param bound_type: str, "up" or "down" or "both"
        :return: first_exceeding_time
        """
        bound_star = np.log(bound/self.s0)/self.sig
        index = self.bm.max_index_in_boundary(bound=bound_star, bound_type=bound_type)
        tau = index*self.bm.dt
        return tau, index

    def analytical_first_exceeding_time_pdf(self, bound, tau):
        """
        pdf of first passage time given bound (one side)
        :param bound: int
        :param tau: first passage time
        :return: pdf(tau)
        """
        bound_star = np.log(bound/self.s0)/self.sig
        pdf = self.bm.analytical_first_exceeding_time_pdf(bound=bound_star, tau=tau)
        return pdf

    def analytical_first_exceeding_time_cdf(self, bound, tau):
        """
        cdf of first passage time given bound (one side)
        :param bound: int
        :param tau: first passage time
        :return: cdf(tau)
        """
        bound_star = np.log(bound/self.s0)/self.sig
        cdf = self.bm.analytical_first_exceeding_time_cdf(bound=bound_star, tau=tau)
        return cdf

    def value_cdf(self, t, price):
        value = np.log(price/self.s0)
        return self.bm.value_cdf(t, value)

    def value_pdf(self, t, price):
        value = np.log(price/self.s0)
        return self.bm.value_pdf(t, value)
