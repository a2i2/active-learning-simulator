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
    """
    @abstractmethod
    def initialise(self, sample):
        pass

    @abstractmethod
    def stopping_criteria(self, sample):
        """
        Returns if the active learning should be terminated, i.e. a stopping criterion has been reached

        :param sample: newly selected sample to evaluate
        :return: True if the AL training should be stopped early
        """
        stop = False
        return stop

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_eval_metrics(self):
        pass


class SampleSize(Stopper):
    """
    Stops active learning when the sample does not produce any more relevant documents. Naive approach.

    Attributes:

    - verbose: when true, outputs progress
    """
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

    def get_eval_metrics(self):
        return []

    def verbose_output(self, stop):
        if stop:
            print('Stopping criteria reached: screening sample size is 0')
        return


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
    def __init__(self, N, tau_target, verbose=True):
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
        # baseline inclusion rate
        labels = sample['y']
        BIR = sum(labels) / len(sample)
        # estimate total number of relevants
        self.r_total = BIR * self.N
        self.stopping_criteria(sample)
        return

    def stopping_criteria(self, sample):
        self.k = sum(sample['y'])
        self.r_AL = self.r_AL + self.k
        stop = (self.r_AL >= self.tau_target * self.r_total)
        self.out()
        return stop

    def reset(self):
        return

    def get_eval_metrics(self):
        return []

    def verbose_output(self):
        # recall = 0
        # if self.r_total != 0:
        #    recall = self.r_AL / self.r_total
        # print('Recall:', recall, 'for', self.r_total, 'estimated total relevants')
        print("\r" + 'Number of relevants seen in sample:', str(self.k), 'for a total of', str(self.r_AL), end='')
        #time.sleep(1)
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
        """
        Initialise the stopper
        :param sample: sample data to evaluate
        """
        self.stopping_criteria(sample)
        return

    def stopping_criteria(self, sample):
        """
        Determines if the active learning should be terminated depending on the calculated p-value.

        :param sample: sample data to evaluate
        :return: True if AL should cease
        """
        stop = False
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
            if p < 1 - self.alpha:
                stop = True
                self.alpha = self.alpha_stage_2
        self.out()
        # self.estimate_progress()
        return stop

    def estimate_progress(self):
        """
        WIP
        Estimate the number of documents that should be screened in order to find all relevant documents.
        """
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
        """
        Resets stopper parameters and variables

        :return:
        """
        self.N_s = self.N
        self.r_AL = 0
        self.k = 0
        self.Ys = []
        self.ps = []
        self.K_hats = []
        self.n_est = []
        return

    def get_eval_metrics(self):
        return [{'name': 'p-values', 'x': ('iterations', range(len(self.ps))), 'y': ('p-values', self.ps)}]

    def verbose_output(self):
        """
        Provides verbose outputting for stopper when enabled.
        """
        print("\r" + 'Number of relevants seen in sample:', str(self.k), 'for a total of', str(self.r_AL), end='')
        return


class Ensemble(Stopper):
    """
    Uses hyper geometric sampling to estimate a p-value for stopping criteria, and compares to a predicted proportion

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
        # statistical method
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
        # sample proportion
        self.r_total = 0
        # verbosity
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initialise(self, sample):
        """
        Initialise the stopper
        :param sample: sample data to evaluate
        """
        # baseline inclusion rate
        labels = sample['y']
        BIR = sum(labels) / len(sample)
        # estimate total number of relevants
        self.r_total = BIR * self.N
        self.stopping_criteria(sample)
        return

    def stopping_criteria(self, sample):
        """
        Determines if the active learning should be terminated depending on the calculated p-value.

        :param sample: sample data to evaluate
        :return: True if AL should cease
        """
        stop = False
        self.k = 0
        # calculate p-value for every addition of sample instances
        for i in range(len(sample)):
            y = sample.iloc[i]['y']
            self.k += y
            self.r_AL += + y
            self.N_s -= 1

            self.Ys.append(y)
            Xs = np.cumsum(self.Ys[::-1])
            ns = np.arange(len(self.Ys))

            self.K_hats = np.ceil(self.r_AL / self.tau_target) - (self.r_AL - Xs)
            p = hypergeom.cdf(Xs, self.N_s + ns, self.K_hats, ns + 1).min()
            self.ps.append(p)
            if p < 1 - self.alpha:
                stop = True
                self.alpha = self.alpha_stage_2
        stop *= (self.r_AL >= self.tau_target * self.r_total)
        self.out()
        return stop

    def reset(self):
        """
        Resets stopper parameters and variables

        :return:
        """
        self.N_s = self.N
        self.r_AL = 0
        self.k = 0
        self.Ys = []
        self.ps = []
        self.K_hats = []
        self.n_est = []
        self.r_total = 0
        return

    def get_eval_metrics(self):
        return [{'name': 'p-values', 'x': ('iterations', range(len(self.ps))), 'y': ('p-values', self.ps)}]

    def verbose_output(self):
        """
        Provides verbose outputting for stopper when enabled.
        """
        print("\r" + 'Number of relevants seen in sample:', str(self.k), 'for a total of', str(self.r_AL), end='')
        return

