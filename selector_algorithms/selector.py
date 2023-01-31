from abc import ABC, abstractmethod


class Selector(ABC):
    """
    Base selector_algorithms abstract class that provides base functionality requirements for sampling instances for active learning

    Methods:

    - initial_select: Provides the initial sample selection from a dataset
    - select: Selects instances for active learning based on a selection criteria
    - reset: Resets the selector_algorithms

    Attributes:

    - batch_size: number of documents screened per iteration
    - verbose: specifies the level of verbosity, True or False
    """
    def __init__(self, batch_size, verbose=False):
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

    def get_eval_metrics(self):
        """
        Provides metrics specific to the selector_algorithms for later visualisation (optional).

        :return: list of metrics. Format for metric:
            metrics = [{'name': plot_name, 'x': (x_label, x_values), 'y': (y_label, y_values)}, ...]
        """
        return None
