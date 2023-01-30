import statistics
from abc import ABC, abstractmethod

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from scipy.stats import lognorm

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

    @abstractmethod
    def reset(self, **params):
        """
        Reset model parameters to their initial state

        :param params: arguments for recreation of the model
        """
        pass

    @abstractmethod
    def get_eval_metrics(self):
        """
        Provides metrics specific to the model for later visualisation (optional).

        :return: list of metrics. Format for metric:
            metrics = [{'name': plot_name, 'x': (x_label, x_values), 'y': (y_label, y_values)}, ...]
        """
        pass


class NB(Model):
    def __init__(self, params=None):
        self.model = BernoulliNB()

    def train(self, X, Y):
        self.model.fit(X, Y)
        return

    def test(self, X, Y):
        y_prob = self.model.predict_proba(X)
        return y_prob

    def predict(self, X, Y):
        y_pred = self.model.predict(X)
        return y_pred

    def reset(self, params=None):
        self.model = BernoulliNB()
        return

    def get_eval_metrics(self):
        return


class LR(Model):
    def __init__(self, params=None):
        self.model = LogisticRegression()

    def train(self, X, Y):
        self.model.fit(X, Y)
        return

    def test(self, X, Y):
        y_prob = self.model.predict_proba(X)
        return y_prob

    def predict(self, X, Y):
        y_pred = self.model.predict(X)
        return y_pred

    def reset(self, params=None):
        self.model = LogisticRegression()
        return

    def get_eval_metrics(self):
        return


class SVC(Model):
    def __init__(self, params=None):
        self.model = LinearSVC(params)

    def train(self, X, Y):
        self.model.fit(X, Y)
        return

    def test(self, X, Y):
        y_prob = self.model.decision_function(X)
        y_prob = np.vstack((y_prob, y_prob)).T
        return y_prob

    def predict(self, X, Y):
        y_pred = self.model.predict(X)
        return y_pred

    def reset(self, params=None):
        self.model = LinearSVC(params)
        return

    def get_eval_metrics(self):
        return


class MLP(Model):
    def __init__(self, params=None):
        self.model = MLPClassifier(params)

    def train(self, X, Y):
        self.model.fit(X, Y)
        return

    def test(self, X, Y):
        y_prob = self.model.predict_proba(X)
        return y_prob

    def predict(self, X, Y):
        y_pred = self.model.predict(X)
        return y_pred

    def reset(self, params=None):
        self.model = MLPClassifier(params)
        return

    def get_eval_metrics(self):
        return


class Ideal(Model):
    def __init__(self, params=None):
        pass

    def train(self, X, Y):
        return

    def test(self, X, Y):
        y_prob = list(Y)
        y_prob = np.vstack((y_prob, y_prob)).T
        return y_prob

    def predict(self, X, Y):
        y_pred = Y
        return y_pred

    def reset(self, params=None):
        return

    def get_eval_metrics(self):
        return


"""=================================================================================================================="""


class NewModel(Model):
    """
    Abstract class that provides the base functionality requirements for a machine learning model

    Methods:

    - train: Trains machine learning model
    - test: Computes the class likelihoods for each test instance in the dataset
    - predict: Predicts the label for each instance in the testing dataset
    - score: Evaluates model on testing dataset
    - reset: Reset model parameters
    """
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self.scores = []
        pass

    def train(self, X, Y):
        """
        Trains machine learning model

        :param Y: training dataset labels 'y'
        :param X: training dataset features 'x'
        """
        self.model.fit(X, Y)
        pass

    def test(self, X, Y):
        """
        Computes the class likelihoods for each test instance in the dataset

        :param Y: testing dataset labels 'y'
        :param X: testing dataset features 'x'
        :return: list of probability predictions for each class for each test instance, classes: [0, 1]
        """
        y_prob = self.model.predict_proba(X)
        self.scores.append(self.model.score(X, Y))
        return y_prob

    def predict(self, X, Y):
        """
        Predicts the label for each instance in the testing dataset

        :param Y: testing dataset labels 'y'
        :param X: testing dataset features 'x'
        :return: label predictions for each instance
        """
        y_pred = self.model.predict(X)
        return y_pred

    def reset(self):
        """
        Reset model parameters to their initial state

        """
        self.model = DecisionTreeClassifier()
        pass

    def get_eval_metrics(self):
        """
        Provides metrics specific to the model for later visualisation (optional).

        :return: list of metrics. Format for metric:
            metrics = [{'name': plot_name, 'x': (x_label, x_values), 'y': (y_label, y_values)}, ...]
        """
        metric = [{'name': 'model', 'x': ('iterations', list(range(len(self.scores)))), 'y': ('model acc', self.scores)}]
        return metric
