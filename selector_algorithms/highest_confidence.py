import numpy as np

from selector_algorithms.selector import Selector


class HighestConfidence(Selector):
    """
    Highest Confidence Selector
    Basic selector_algorithms that gets samples based on ML prediction confidence

    Attributes:

    - batch_size: size of each selection / sample
    - p_threshold: confidence level threshold to discriminate valid instances for a sample (unused)
    """
    def __init__(self, batch_size, verbose=False):
        super().__init__(batch_size, verbose)
        self.batch_size = batch_size
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initial_select(self, data, data_indices):
        # randomly sample from test set to be training instances
        sample = np.random.choice(data_indices, min(self.batch_size, data_indices.size), replace=False)
        self.out(sample)
        return sample

    def select(self, test_indices, predictions):
        # simple: take highest confidence ones
        sample = predictions[:, 1].argsort()[::-1]  # reverse order?
        # sample = predictions[:, 0].argsort()# alternative?
        # exclude those outside of the threshold
        # sorted_preds = predictions[:, 1][sample]
        # sample = sample[sorted_preds >= self.p_threshold]
        # only take a subset of the sample, get the actual instance indices
        sample = test_indices[sample[0: min(self.batch_size, sample.size)]]
        self.out(sample)
        return sample

    def verbose_output(self, sample):
        print("\r" + 'Screening ' + str(sample.size) + " instances", end='')
