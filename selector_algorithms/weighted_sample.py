import numpy as np

from selector_algorithms.selector import Selector


class WeightedSample(Selector):
    def __init__(self, batch_size, verbose=False):
        super().__init__(batch_size, verbose)
        self.batch_size = batch_size
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initial_select(self, data, data_indices):
        # randomly sample from test set to be training instances
        sample_indices = np.random.choice(data_indices, min(self.batch_size, data_indices.size), replace=False)
        return sample_indices

    def select(self, test_indices, predictions):
        # random sample, weight higher confidence predictions more
        sample = np.random.choice(test_indices, self.batch_size, replace=False,
                                  p=(predictions / predictions.sum(axis=0, keepdims=1)))
        self.out(sample)
        return sample

    def verbose_output(self, sample):
        print("\r" + 'Screening ' + str(sample.size) + " instances", end='')
