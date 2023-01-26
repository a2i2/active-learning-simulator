import importlib
import math
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import hypergeom


class Stopper(ABC):
    """
    Abstract class that provides base functionality requirements for the stopping criteria object

    Methods:

    - initialise: called during the initial sampling of the active learner
    - stopping_criteria: determines if a particular criteria for stopping has been met from the newly selected sample, called each iteration of the AL training
    - reset: resets any parameters and variables

    Attributes:

    - stop: specifics status of stopping criteria: 1 -> stop active learning, 0 -> continue active learning, -1 -> commence random learning
    - N: number of total documents in the dataset
    - confidence: confidence level parameter
    - verbose: specifics level of verbosity, True or False
    """
    def __init__(self, N, confidence, verbose=False):
        self.stop = False
        self.N = N
        self.confidence = confidence
        self.verbose = verbose
        pass

    @abstractmethod
    def initialise(self, sample):
        """
        Updates the stopper for the initial samples: called each iteration of initial sampling.

        :param sample:
        :return:
        """
        pass

    @abstractmethod
    def stopping_criteria(self, sample):
        """
        Returns if the active learning should be terminated, i.e. a stopping criterion has been reached or whether
        random learning should begin instead.

        :param sample: newly selected sample to evaluate
        :return: 1 if the AL training should be stopped early, 0 if AL training should continue, -1 if random training should begin
        """
        stop = False
        return stop

    @abstractmethod
    def reset(self):
        """
        Resets stopper to initial state.

        :return:
        """
        self.stop = False
        pass

    @abstractmethod
    def get_eval_metrics(self):
        """
        Provides metrics specific to the stopper for later visualisation (optional).

        :return:
        """
        pass


class SampleProportion(Stopper):
    """
    Determines an estimate proportion of relevant documents in the dataset. Calculates the number of relevants documents
    required to be screened; AL stops when (most) relevant documents are thought to be screened.

    Attributes:

    - tau_target: target recall
    - r_total: total number of relevant documents in the dataset
    - r_AL: current number of relevant documents screened by the active learner
    - N: total number of documents in the dataset
    - verbose: when true, outputs progress
    """
    def __init__(self, N, tau_target=0.95, verbose=True):
        super().__init__(N, tau_target, verbose)
        self.n_sample = 0
        self.r_sample = 0
        self.tau_target = tau_target
        self.r_total = 0
        self.r_AL = 0
        self.k = 0
        self.N = N
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initialise(self, sample):
        self.n_sample += len(sample)
        self.r_sample += sum(sample['y'])
        # baseline inclusion rate
        BIR = self.r_sample / self.n_sample
        # estimate total number of relevants
        self.r_total = BIR * self.N
        self.stopping_criteria(sample)
        return

    def stopping_criteria(self, sample):
        self.k = sum(sample['y'])
        self.r_AL = self.r_AL + self.k
        self.stop = (self.r_AL >= self.tau_target * self.r_total)
        self.out()
        return

    def reset(self):
        self.n_sample = 0
        self.r_sample = 0
        return

    def get_eval_metrics(self):
        return []

    def verbose_output(self):
        print("\r" + 'Number of relevants seen in sample:', str(self.k), 'for a total of', str(self.r_AL), end='')
        return


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
        self.tau_target = tau_target
        self.N = N
        self.N_s = self.N
        self.r_AL = 0
        self.k = 0
        self.alpha_stage_1 = alpha / 2
        self.alpha_stage_2 = alpha
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

    def reset(self):
        self.N_s = self.N
        self.r_AL = 0
        self.k = 0
        self.Ys = []
        self.ps = []
        self.K_hats = []
        self.n_est = []
        return

    def get_eval_metrics(self):
        return [{'name': 'stopper', 'x': ('iterations', list(range(len(self.ps)))), 'y': ('p-values', self.ps)}]

    def verbose_output(self):
        print("\r" + 'Number of relevants seen in sample:', str(self.k), 'for a total of', str(self.r_AL), end='')
        return


class ConsecutiveCount(Stopper):
    """
    Ceases active learning when consecutive number of irrelevant documents are seen

    Attributes:

    - N: total number of documents in the full dataset
    - threshold: maximum number of allowable irrelevant documents
    - count: number of current irrelevant documents seen consecutively
    - verbose: when true, outputs progress
    """
    def __init__(self, N, tau_target=0.95, threshold=0.05, verbose=False):
        super().__init__(N, tau_target, verbose)
        self.N = N
        self.threshold = float(threshold) * self.N
        self.count = 0
        self.counts = []
        self.k = 0
        self.r_AL = 0
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initialise(self, sample):
        self.count = 0
        return

    def stopping_criteria(self, sample):
        self.k = sum(sample['y'])
        self.r_AL += self.k
        self.stop = False
        # calculate p-value for every addition of sample instances
        for i in range(len(sample)):
            y = sample.iloc[i]['y']
            self.count = (not y) * (self.count + 1)
            self.counts.append(self.count)
            if self.count >= self.threshold:
                self.stop = True
                break
        return

    def reset(self):
        self.count = 0
        self.k = 0
        return

    def get_eval_metrics(self):
        return [{'name': 'stopper', 'x': ('iterations', list(range(len(self.counts)))), 'y': ('irrelevant counts', self.counts)}]

    def verbose_output(self):
        print("\r" + 'Number of relevants seen in sample:', str(self.k), 'for a total of', str(self.r_AL), end='')
        return


class Ensembler(Stopper):
    """

    """
    def __init__(self, N, tau_target=0.95, *stoppers, verbose=False):
        super().__init__(N, tau_target, verbose)
        self.stoppers = []
        for stopper in stoppers:
            self.stoppers.append(globals()[stopper](N, tau_target))
        # verbosity
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initialise(self, sample):
        for stopper in self.stoppers:
            stopper.initialise(sample)
        return

    def stopping_criteria(self, sample):
        self.stop = False
        for stopper in self.stoppers:
            self.stop *= stopper.stopping_criteria(sample)
        return self.stop

    def reset(self):
        for stopper in self.stoppers:
            stopper.reset()
        return

    def get_eval_metrics(self):
        metrics = []
        for stopper in self.stoppers:
            metrics.append(*stopper.get_eval_metrics())
        return metrics

    def verbose_output(self):
        for stopper in self.stoppers:
            stopper.verbose_output()
        return
