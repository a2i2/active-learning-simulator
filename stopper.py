import math
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import hypergeom


class Stopper(ABC):
    @abstractmethod
    def initialise(self, sample):
        pass

    @abstractmethod
    def stopping_criteria(self, sample):
        pass

    @abstractmethod
    def reset(self):
        pass


class SampleSize(Stopper):
    def __init__(self, verbose=True):
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initialise(self, sample):
        return

    def stopping_criteria(self, sample):
        stop = len(sample) == 0
        self.out(stop)
        return stop

    def reset(self):
        return

    def verbose_output(self, stop):
        if stop:
            print('Stopping criteria reached: screening sample size is 0')
        return


class SampleProportion(Stopper):
    def __init__(self, N, tau_target, verbose=True):
        self.tau_target = tau_target
        self.r_total = 0
        self.r_AL = 0
        self.N = N
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initialise(self, sample):
        # baseline inclusion rate
        labels = sample['y']
        BIR = sum(labels) / len(sample)
        # estimate total number of relevants
        self.r_total = BIR * self.N
        self.stopping_criteria(sample)
        return

    def stopping_criteria(self, sample):
        k = sum(sample['y'])
        self.r_AL = self.r_AL + k
        stop = (self.r_AL >= self.tau_target * self.r_total)
        self.out()
        return stop

    def reset(self):
        return

    def verbose_output(self):
        recall = 0
        if self.r_total != 0:
            recall = self.r_AL / self.r_total
        print('Recall:', recall, 'for', self.r_total, 'estimated total relevants')
        print('Number of relevants seen:', self.r_AL)
        return


class Statistical(Stopper):
    def __init__(self, N, tau_target, alpha=0.95, verbose=False):
        self.tau_target = tau_target
        self.N = N
        self.N_s = self.N
        self.r_AL = 0
        self.k = 0
        self.alpha = alpha
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
        stop = False
        for i in range(len(sample)):
            y = sample.iloc[i]['y']
            self.r_AL = self.r_AL + y
            self.N_s -= 1

            self.Ys.append(y)
            Xs = np.cumsum(self.Ys[::-1])
            ns = np.arange(len(self.Ys))

            self.K_hats = np.ceil(self.r_AL / self.tau_target) - (self.r_AL - Xs)
            p = hypergeom.cdf(Xs, self.N_s + ns, self.K_hats, ns + 1).min()
            self.ps.append(p)
            if p < 1 - self.alpha:
                stop = True
        self.out()
        # self.estimate_progress()
        return stop

    def estimate_progress(self):
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

    def reset(self):
        self.N_s = self.N
        self.r_AL = 0
        self.k = 0
        self.Ys = []
        self.ps = []
        self.K_hats = []
        self.n_est = []
        return

    def verbose_output(self):
        print('Number of relevants seen:', self.r_AL)
        return