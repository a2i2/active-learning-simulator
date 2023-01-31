import math

import numpy as np
from scipy.stats import hypergeom

from stopper_algorithms.stopper import Stopper


class Statistical(Stopper):
    """
    Uses hypergeometric sampling to estimate a p-value for stopping criteria.

    Attributes:

    - tau_target: target recall
    - N: total number of documents in the full dataset
    - N_s: total number of documents left in the dataset
    - r_AL: current number of relevant documents screened by the active learner
    - k: number of relevant documents in the sample
    - alpha: confidence level
    - Ys: screening results during training
    - ps: p-values during training
    - K_hats: estimate K values during training
    - n_est: estimate number of documents to be screened during training
    - verbose: when true, outputs progress
    """
    def __init__(self, N, tau_target=0.95, alpha=0.95, verbose=False):
        super().__init__(N, tau_target, verbose)
        self.tau_target = float(tau_target)
        self.N = N
        self.N_s = self.N
        self.r_AL = 0
        self.k = 0
        self.alpha_stage_1 = float(alpha) / 2
        self.alpha_stage_2 = float(alpha)
        self.alpha = self.alpha_stage_1
        self.Ys = []
        self.ps = []
        self.K_hats = []
        self.n_est = []
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initialise(self, sample):
        self.stopping_criteria(sample)
        return

    def stopping_criteria(self, sample):
        self.stop = False
        self.k = 0
        # calculate p-value for every addition of sample instances
        for i in range(len(sample)):
            y = sample.iloc[i]['y']
            self.k += y
            self.r_AL = self.r_AL + y
            self.N_s -= 1

            self.Ys.append(y)
            Xs = np.cumsum(self.Ys[::-1])
            ns = np.arange(len(self.Ys))

            self.K_hats = np.ceil(self.r_AL / self.tau_target) - (self.r_AL - Xs)
            p = hypergeom.cdf(Xs, self.N_s + ns, self.K_hats, ns + 1).min()
            self.ps.append(p)
            # adjust alpha for random sampling
            if p < 1 - self.alpha:
                self.stop = -1
                self.alpha = self.alpha_stage_2
        self.out()
        return

    def estimate_progress(self):
        # WIP
        N = self.N_s
        if N == 0:
            self.n_est.append(0)
            return
        z = 0.95
        p = 1 - self.ps[-1]  # self.K_hats[-1] / N
        q = 1 - p
        E = 1 - z
        numer = N * z ** 2 * p * q
        denom = E ** 2 * (N - 1) + z ** 2 * p * q
        n = math.ceil(numer / denom)
        # n_batch = math.ceil(n / self.batch_size)
        self.n_est.append(n)
        print(n, self.K_hats[-1])
        return

    def get_eval_metrics(self):
        return [{'name': 'stopper', 'x': ('iterations', list(range(len(self.ps)))), 'y': ('p-values', self.ps)}]

    def verbose_output(self):
        print("\r" + 'Number of relevants seen in sample:', str(self.k), 'for a total of', str(self.r_AL), end='')
        return
