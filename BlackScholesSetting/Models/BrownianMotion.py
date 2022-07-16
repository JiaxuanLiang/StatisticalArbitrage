# -*- coding: utf-8 -*- 
"""
--------------------------------------------------
File Name:        BrownianMotion
Description:    
Author:           jiaxuanliang
Date:             7/10/22
--------------------------------------------------
Change Activity:  7/10/22
--------------------------------------------------
"""

import numpy as np
from scipy.stats import norm
import utils


class BM:
    """
    Description: Model of Brownian motion with drift, Xt = ut+Wt,
                 set mu=0 to have standard Brownian motion,
                 also take drift, seed when initializing
    Method:
    simulate_path: Monte Carlo use one generator - np.random.default_rng(seed=seed)
                   It returns different path at calling every time
                   Can save one record of simulation by setting update=True when calling object.simulate_path,
                   and access later by object.path (the first time it always saves)
                   Can access the most recent simulated path by object.last_simulated_path
    largest_index_in_boundary: return the last index of saved path or given path ('up' or 'down' or 'both')
                   that still falls in the boundary
    analytical_first_exceeding_time_pdf: return the analytical first exceeding time pdf
                   (bound and mu have same direction)
    analytical_first_exceeding_time_pdf: return the analytical first exceeding time cdf
                   (bound and mu have same direction)
    """
    def __init__(self, drift, seed):
        self.mu = drift
        self.rng = np.random.default_rng(seed=seed)
        self.path = None
        self.dt = None
        self.last_simulated_path = None
        self.last_simulated_dt = None

    def simulate_path(self, shape, dt, update):
        """
        monte carlo simulation
        :param shape: int, or tuple in the form of (n_simulation, n_time_step)
        :param dt: step length, time interval
        :param update: bool, whether to save the simulated path
                       To have stable result, turn to True and access through object.read_path()
        :return: paths
        """
        x0 = 0
        dw = self.rng.standard_normal(shape)

        if len(dw.shape) == 1:
            x = x0+np.cumsum(self.mu*dt+np.sqrt(dt)*dw)
            self.last_simulated_path = np.append(x0, x)
            self.last_simulated_dt = dt
            if any([update, self.path is None]):
                self.path = self.last_simulated_path.copy()
                self.dt = dt
            return self.last_simulated_path

        if len(dw.shape) == 2:
            x = x0+np.cumsum(self.mu*dt+np.sqrt(dt)*dw, axis=1)
            self.last_simulated_path = np.hstack([np.ones(shape[0]).reshape(shape[0], 1)*x0, x])
            self.last_simulated_dt = dt
            if any([update, self.path is None]):
                self.path = self.last_simulated_path.copy()
                self.dt = dt
            return self.last_simulated_path

        if len(dw.shape) not in [1, 2]:
            raise ValueError("tuple should be in the form of (n_simulation, n_time_step)")

    def max_index_in_boundary(self, bound, bound_type, path=None):
        """
        if not path given, use self.path: the first simulated or the most recent update
        :param bound: int, float or tuple
        :param bound_type: str, "up" or "down" or "both"
        :param path: for given path
        :return: the max index of the path within boundary
        """
        up = np.inf
        down = -np.inf

        if path is not None:
            x_path = path.copy()
        else:
            x_path = self.path.copy()

        if bound_type == 'up' and len(np.array(bound).shape) == 0:
            up = bound
            down = -np.inf
        if bound_type == 'down' and len(np.array(bound).shape) == 0:
            up = np.inf
            down = bound
        if bound_type == 'both' and len(np.array(bound).shape) == 1:
            up = max(bound)
            down = min(bound)

        return utils.get_index_or_indexes(x_path, up, down)

    def analytical_first_exceeding_time_pdf(self, bound, tau):
        param_mu = bound/self.mu
        param_lbd = bound**2
        pdf = np.sqrt(param_lbd/(2*np.pi*tau**3))*np.exp(-param_lbd*(tau-param_mu)**2/(2*param_mu**2*tau))
        return pdf

    def analytical_first_exceeding_time_cdf(self, bound, tau):
        # inverse gaussian
        param_mu = bound/self.mu
        param_lbd = bound**2
        item1 = norm.cdf(np.sqrt(param_lbd/tau)*(tau/param_mu-1))
        item2 = np.exp(2*param_lbd/param_mu)*norm.cdf(-np.sqrt(param_lbd/tau)*(tau/param_mu+1))
        cdf = item1 + item2
        return cdf

    def value_cdf(self, t, value):
        z = value-self.mu*t
        return norm.cdf(z)

    def value_pdf(self, t, value):
        z = value-self.mu*t
        return norm.pdf(z)
