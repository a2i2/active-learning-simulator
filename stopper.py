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
    """
    def __init__(self, N, confidence):
        self.stop = False
        self.N = N
        self.confidence = confidence
        pass

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
        self.stop = False
        pass

    @abstractmethod
    def get_eval_metrics(self):
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
        super().__init__(N, tau_target)
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
        self.stop = (self.r_AL >= self.tau_target * self.r_total)
        self.out()
        return

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
        super().__init__(N, tau_target)
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
        return [{'name': 'p-values', 'x': ('iterations', list(range(len(self.ps)))), 'y': ('p-values', self.ps)}]

    def verbose_output(self):
        """
        Provides verbose outputting for stopper when enabled.
        """
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
        super().__init__(N, tau_target)
        self.N = N
        self.threshold = threshold * self.N
        self.count = 0
        self.counts = []
        self.k = 0
        self.r_AL = 0
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initialise(self, sample):
        """
        Initialise the stopper
        :param sample: sample data to evaluate
        """
        self.count = 0
        return

    def stopping_criteria(self, sample):
        """
        Determines if the active learning should be terminated depending on the current number of consecutive
        irrelevant documents seen

        :param sample: sample data to evaluate
        :return: True if AL should cease
        """
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
        """
        Resets stopper parameters and variables

        :return:
        """
        self.count = 0
        self.k = 0
        return

    def get_eval_metrics(self):
        return [{'name': 'consecutive irrelevant count', 'x': ('iterations', range(len(self.counts))), 'y': ('irrelevant counts', self.counts)}]

    def verbose_output(self):
        """
        Provides verbose outputting for stopper when enabled.
        """
        print("\r" + 'Number of relevants seen in sample:', str(self.k), 'for a total of', str(self.r_AL), end='')
        return


class Ensembler(Stopper):
    """

    """
    def __init__(self, N, tau_target=0.95, *stoppers, verbose=False):
        super().__init__(N, tau_target)
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
        """
        Resets stopper parameters and variables

        :return:
        """
        for stopper in self.stoppers:
            stopper.reset()
        return

    def get_eval_metrics(self):
        metrics = []
        for stopper in self.stoppers:
            metrics.append(*stopper.get_eval_metrics())
        return metrics

    def verbose_output(self):
        """
        Provides verbose outputting for stopper when enabled.
        """
        for stopper in self.stoppers:
            stopper.verbose_output()
        return
