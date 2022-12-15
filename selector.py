from abc import ABC, abstractmethod
import numpy as np


class Selector(ABC):
    """
    Base selector abstract class that provides base functionality requirements for sampling instances for active learning

    Methods:

    - initial_select: Provides the initial sample selection from a dataset
    - select: Selects instances for active learning based on a selection criteria
    - reset: Resets the selector
    """
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
        Resets the selection algorithm
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
    def __init__(self, batch_size, p_threshold, verbose=False):
        self.batch_size = batch_size
        self.p_threshold = p_threshold
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initial_select(self, data, data_indices):
        """
        Randomly samples from data set

        :param data: raw dataset
        :param data_indices: corresponding indices for the dataset
        :return: indices of the sample instances
        """
        # randomly sample from test set to be training instances
        sample_indices = np.random.choice(data_indices, self.batch_size, replace=False)
        return sample_indices

    def select(self, test_indices, predictions):
        """
        Selects sample based on ML model confidence in prediction (class probability)

        :param test_indices: indices of the available instances
        :param predictions: corresponding ML predictions for each label for each test instance
        :return: indices of the sample instances
        """
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


class LowestEntropy(Selector):
    def __init__(self, batch_size, p_threshold, verbose=False):
        self.batch_size = batch_size
        self.p_threshold = p_threshold
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initial_select(self, data, data_indices):
        # randomly sample from test set to be training instances
        sample_indices = np.random.choice(data_indices, self.batch_size, replace=False)
        return sample_indices

    def select(self, test_indices, predictions):
        """
        Selects sample based on ML model entropy in prediction, chooses relevant*? instances with the lowest entropy

        :param test_indices: indices of the available instances
        :param predictions: corresponding ML predictions for each label for each test instance
        :return: indices of the sample instances
        """
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
        print('Screening ' + str(sample.size) + " instances")


class WeightedSample(Selector):
    def __init__(self, batch_size, verbose=False):
        self.batch_size = batch_size
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initial_select(self, data, data_indices):
        # randomly sample from test set to be training instances
        sample_indices = np.random.choice(data_indices, self.batch_size, replace=False)
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
        print('Screening ' + str(sample.size) + " instances")


# clusterer = AgglomerativeClustering(n_clusters=batch_size)
# selector = ClusterSelector(clusterer, data['train']['x'].apply(pd.Series), batch_size, p_threshold)
class Cluster(Selector):
    def __init__(self, clusterer, data, batch_size, p_threshold, verbose=False):
        self.p_threshold = p_threshold
        self.batch_size = batch_size
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None
        self.clusterer = clusterer
        self.clusters = self.create_clusters(data)

    def create_clusters(self, data):
        clusters = self.clusterer.fit(data)
        print(clusters)
        return clusters.labels_

    def initial_select(self, data, data_indices):
        return

    def select(self, test_indices, predictions):
        # simple: take highest confidence ones
        sample = predictions.argsort()
        sorted_preds = predictions[sample]
        # exclude those outside of the threshold
        sample = sample[sorted_preds >= self.p_threshold]
        # only take a subset of the sample, get the actual instance indices
        sample = test_indices[sample[0: min(self.batch_size, sample.size)]]
        self.out(sample)
        return sample

    def reset(self):
        return

    def verbose_output(self, sample):
        print('Screening ' + str(sample.size) + " instances")
