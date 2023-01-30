from abc import ABC, abstractmethod
import numpy as np


class Selector(ABC):
    """
    Base selector abstract class that provides base functionality requirements for sampling instances for active learning

    Methods:

    - initial_select: Provides the initial sample selection from a dataset
    - select: Selects instances for active learning based on a selection criteria
    - reset: Resets the selector

    Attributes:

    - batch_size: number of documents screened per iteration
    - verbose: specifies the level of verbosity, True or False
    """
    def __init__(self, batch_size, verbose):
        self.batch_size = batch_size
        self.verbose = verbose
        pass

    @abstractmethod
    def initial_select(self, data, data_indices):
        """
        Provides the initial sample selection from a dataset

        :param data: raw dataset
        :param data_indices: corresponding indices for the dataset
        :return: indices of the sample instances
        """
        pass

    @abstractmethod
    def select(self, test_indices, predictions):
        """
        Using ML model predictions to sample from the available instances

        :param test_indices: indices of the available instances
        :param predictions: corresponding ML predictions for each label for each test instance
        :return: indices of the sample instances
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the selection algorithm to its initial state
        """
        pass

    @abstractmethod
    def get_eval_metrics(self):
        """
        Provides metrics specific to the selector for later visualisation (optional).

        :return: list of metrics. Format for metric:
            metrics = [{'name': plot_name, 'x': (x_label, x_values), 'y': (y_label, y_values)}, ...]
        """
        pass


class HighestConfidence(Selector):
    """
    Highest Confidence Selector
    Basic selector that gets samples based on ML prediction confidence

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

    def reset(self):
        return

    def verbose_output(self, sample):
        print("\r" + 'Screening ' + str(sample.size) + " instances", end='')

    def get_eval_metrics(self):
        return


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

    def reset(self):
        return

    def verbose_output(self, sample):
        print("\r" + 'Screening ' + str(sample.size) + " instances", end='')

    def get_eval_metrics(self):
        return


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

    def reset(self):
        return

    def verbose_output(self, sample):
        print("\r" + 'Screening ' + str(sample.size) + " instances", end='')

    def get_eval_metrics(self):
        return


"""=================================================================================================================="""


class NewSelector(Selector):
    """
    Methods:

    - initial_select: Provides the initial sample selection from a dataset
    - select: Selects instances for active learning based on a selection criteria
    - reset: Resets the selector

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
        sample = np.random.choice(data_indices, min(self.batch_size, data_indices.size), replace=False)
        self.out()
        return sample

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

    def reset(self):
        """
        Resets the selection algorithm to its initial state
        """
        return

    def verbose_output(self):
        print("\r" + "Hello, I'm screening now", end='')

    def get_eval_metrics(self):
        """
        Provides metrics specific to the selector for later visualisation (optional).

        :return: list of metrics. Format for metric:
            metrics = [{'name': plot_name, 'x': (x_label, x_values), 'y': (y_label, y_values)}, ...]
        """
        return
