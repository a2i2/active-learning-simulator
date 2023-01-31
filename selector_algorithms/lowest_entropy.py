import numpy as np

from selector_algorithms.selector import Selector


class LowestEntropy(Selector):
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
        # simple: take lowest entropy instances
        entropies = self.calc_entropy(predictions)
        sample = entropies.argsort()
        # only take a subset of the sample, get the actual instance indices
        sample = test_indices[sample[0: min(self.batch_size, sample.size)]]
        self.out(sample)
        return sample

    def calc_entropy(self, predictions):
        """
        Calculates the entropy of prediction between each class for each instance

        :param predictions: list of class probability predictions
        :return: list of entropies for each instance
        """
        entropy = -np.sum(predictions * np.log2(predictions), axis=1)
        return entropy

    def verbose_output(self, sample):
        print("\r" + 'Screening ' + str(sample.size) + " instances", end='')
