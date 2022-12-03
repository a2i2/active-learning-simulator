import statistics
from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from scipy.stats import lognorm


class Model(ABC):
    @abstractmethod
    def train(self, train_data):
        """
        Trains machine learning model
        :param train_data: training dataset containing features 'x' and ground truth labels 'y'
        """
        pass

    @abstractmethod
    def test(self, test_data):
        """
        Computes the class likelihoods for each test instance in the dataset
        :param test_data: testing dataset containing features 'x'
        :return: list of probability predictions for each class for each test instance
        """
        pass

    @abstractmethod
    def predict(self, test_data):
        """
        Predicts the label for each instance in the testing dataset
        :param test_data: testing dataset containing features 'x'
        :return: label predictions for each instance
        """
        pass

    @abstractmethod
    def score(self, test_data):
        """
        Evaluates model on testing dataset
        :param test_data: testing dataset containing features 'x' and ground truth labels 'y'
        :return: evaluation metrics
        """
        pass

    @abstractmethod
    def reset(self, **params):
        """
        Reset model parameters
        :param params: arguments for recreation of the model
        """
        pass


class NB(Model):
    def __init__(self, **params):
        self.model = BernoulliNB(**params)

    def train(self, train_data):
        self.model.fit(train_data['x'].apply(pd.Series), train_data['y'])
        return

    def test(self, test_data):
        y_prob = self.model.predict_proba(test_data['x'].apply(pd.Series))
        return y_prob

    def predict(self, test_data):
        y_pred = self.model.predict(test_data['x'].apply(pd.Series))
        return y_pred

    def score(self, test_data):
        score = self.model.score(test_data['x'].apply(pd.Series), test_data['y'])
        return score

    def reset(self, **params):
        self.model = BernoulliNB(**params)
        return


class LR(Model):
    def __init__(self, **params):
        self.model = LogisticRegression(**params)

    def train(self, train_data):
        self.model.fit(train_data['x'].apply(pd.Series), train_data['y'])
        return

    def test(self, test_data):
        y_prob = self.model.predict_proba(test_data['x'].apply(pd.Series))
        return y_prob

    def predict(self, test_data):
        y_pred = self.model.predict(test_data['x'].apply(pd.Series))
        return y_pred

    def score(self, test_data):
        score = self.model.score(test_data['x'].apply(pd.Series), test_data['y'])
        return score

    def reset(self, **params):
        self.model = BernoulliNB(**params)
        return


class SVC(Model):
    def __init__(self, **params):
        self.model = LinearSVC(verbose=True, **params)

    def train(self, train_data):
        self.model.fit(train_data['x'].apply(pd.Series), train_data['y'])
        return

    def test(self, test_data):
        y_prob = self.model.decision_function(test_data['x'].apply(pd.Series))
        y_prob = np.vstack((y_prob, y_prob)).T
        return y_prob

    def predict(self, test_data):
        y_pred = self.model.predict(test_data['x'].apply(pd.Series))
        return y_pred

    def score(self, test_data):
        score = self.model.score(test_data['x'].apply(pd.Series), test_data['y'])
        return score

    def reset(self, **params):
        self.model = LinearSVC(**params)
        return


class MLP(Model):
    def __init__(self, **params):
        self.model = MLPClassifier(**params)

    def train(self, train_data):
        self.model.fit(train_data['x'].apply(pd.Series), train_data['y'])
        return

    def test(self, test_data):
        y_prob = self.model.predict_proba(test_data['x'].apply(pd.Series))
        return y_prob

    def predict(self, test_data):
        y_pred = self.model.predict(test_data['x'].apply(pd.Series))
        return y_pred

    def score(self, test_data):
        score = self.model.score(test_data['x'].apply(pd.Series), test_data['y'])
        return score

    def reset(self, **params):
        self.model = MLPClassifier(**params)
        return


class NaiveBayes(Model):
    def __init__(self, **params):
        self.N = 0
        self.priors_counts = 0
        self.likelihoods = {'mean': 0, 'var': 0}

    def train(self, data):
        self.calc_priors(data)
        self.calc_likelihoods(data)

    def calc_priors(self, data):
        sample_count = sum(data['y'])
        self.priors_counts += sample_count

    def calc_likelihoods(self, data):
        Ns = len(data['y'])
        sample_mean = np.mean(data['x'], axis=1)
        sample_var = np.std(data['x'], axis=1)

        new_N = self.N + Ns
        new_mean = (self.N * self.likelihoods['mean'] + Ns * sample_mean) / new_N

        denom = new_N**2
        new_var = self.N ** 2 / denom * self.likelihoods['var'] + Ns**2 / denom * sample_var

        self.likelihoods = {'mean': new_mean, 'var': new_var}
        self.N = new_N

    def test(self, test_data):

        return

    def lognormpdf(self, x, mean, sd):
        # epsilon smoothing
        epsilon = 1e-9
        sd += epsilon

        var = np.power(sd, 2)
        den = np.sqrt(2 * np.pi * var)
        num = np.exp(-np.power(x - mean, 2) / (2 * var))

        P = num / den
        # check for 0 probabilities
        P += epsilon
        logP = np.log2(P)
        return logP

    def predict(self, test_data):
        return

    def score(self, test_data):

        return

    def reset(self, **params):
        self.N = 0
        self.priors_counts = 0
        self.likelihoods = {'mean': 0, 'var': 0}
        return

