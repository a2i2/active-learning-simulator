
from abc import ABC, abstractmethod


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
        self.stop = False
        return

    def get_eval_metrics(self):
        """
        Provides metrics specific to the stopper for later visualisation (optional).

        :return: list of metrics. Format for metric:
            metrics = [{'name': plot_name, 'x': (x_label, x_values), 'y': (y_label, y_values)}, ...]
        """
        return None
