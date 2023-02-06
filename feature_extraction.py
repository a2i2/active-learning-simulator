from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    """
    Abstract class that provides the base functionality requirements for a machine learning model

    Methods:

    - extract_features: performs the feature extraction on a given dataset
    """

    @abstractmethod
    def extract_features(self, data):
        """
        Performs the feature extraction on a given dataset

        :param data: raw data set containing abstract and title for features and relevant label
        :return: feature extracted dataset with new features
        """
        return data



