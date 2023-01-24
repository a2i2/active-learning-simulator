from abc import ABC, abstractmethod
from sklearn.cluster import AgglomerativeClustering, KMeans
import pandas as pd
import numpy as np


class Clusterer(ABC):
    """
    Abstract class that provides the base functionality requirements for a machine learning model

    Methods:

    - fit: Fit clustering to data
    - predict: Predicts the label for each instance in the testing dataset
    """
    @abstractmethod
    def fit(self, data):
        """
        Trains machine learning model

        :param data: training dataset containing features 'x' and ground truth labels 'y'
        """
        pass

    @abstractmethod
    def predict(self, data):
        """
        Predicts the label for each instance in the testing dataset

        :param data: testing dataset containing features 'x'
        :return: label predictions for each instance
        """
        pass


class KM(Clusterer):

    def __init__(self, arguments):
        self.n_clusters = int(arguments[0])
        self.clusterer = KMeans(n_clusters=self.n_clusters)

    def fit(self, data):
        self.clusterer.fit(data)
        return

    def predict(self, data):
        clusters = self.clusterer.predict(data)
        return clusters
