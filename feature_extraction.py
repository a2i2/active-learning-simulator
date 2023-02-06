from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    """
    Abstract class that provides the base functionality requirements for a machine learning model

    Methods:

    - extract_features:

    """

    @abstractmethod
    def extract_features(self, data):
        """

        :param data:
        :return:
        """
        return data



