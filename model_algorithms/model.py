from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract class that provides the base functionality requirements for a machine learning model

    Methods:

    - train: Trains machine learning model
    - test: Computes the class likelihoods for each test instance in the dataset
    - predict: Predicts the label for each instance in the testing dataset
    - score: Evaluates model on testing dataset
    - reset: Reset model parameters
    """
    @abstractmethod
    def train(self, X, Y):
        """
        Trains machine learning model

        :param Y: training dataset labels 'y'
        :param X: training dataset features 'x'
        """
        pass

    @abstractmethod
    def test(self, X, Y):
        """
        Computes the class likelihoods for each test instance in the dataset

        :param Y: training dataset labels 'y'
        :param X: training dataset features 'x'
        :return: list of probability predictions for each class for each test instance
        """
        pass

    @abstractmethod
    def predict(self, X, Y):
        """
        Predicts the label for each instance in the testing dataset

        :param Y: training dataset labels 'y'
        :param X: training dataset features 'x'
        :return: label predictions for each instance
        """
        pass

    def get_eval_metrics(self):
        """
        Provides metrics specific to the model for later visualisation (optional).

        :return: list of metrics. Format for metric:
            metrics = [{'name': plot_name, 'x': (x_label, x_values), 'y': (y_label, y_values)}, ...]
        """
        return None
