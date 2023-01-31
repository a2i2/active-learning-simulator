import numpy as np

from selector_algorithms.selector import Selector


class NewSelector(Selector):
    """
    Methods:

    - initial_select: Provides the initial sample selection from a dataset
    - select: Selects instances for active learning based on a selection criteria
    - reset: Resets the selector_algorithms

    Attributes:

    - batch_size: number of documents screened per iteration
    - verbose: specifies the level of verbosity, True or False
    """
    def __init__(self, batch_size, verbose):
        super().__init__(batch_size, verbose)
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initial_select(self, data, data_indices):
        """
        Provides the initial sample selection from a dataset

        :param data: raw dataset
        :param data_indices: corresponding indices for the dataset
        :return: indices of the sample instances
        """
        # randomly sample from test set to be training instances
        sample_indices = np.random.choice(data_indices, min(self.batch_size, data_indices.size), replace=False)
        self.out()
        return sample_indices

    def select(self, test_indices, predictions):
        """
        Using ML model predictions to sample from the available instances

        :param test_indices: indices of the available instances
        :param predictions: corresponding ML predictions for each label for each test instance
        :return: indices of the sample instances
        """
        # simple: take the highest confidence ones
        sample_indices = predictions[:, 1].argsort()[::-1]
        # only take a subset of the sample, get the actual instance indices
        sample_indices = test_indices[sample_indices[0: min(self.batch_size, sample_indices.size)]]
        self.out()
        return sample_indices

    def verbose_output(self):
        print("\r" + "Hello, I'm screening now", end='')
