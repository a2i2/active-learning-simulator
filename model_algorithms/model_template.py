from sklearn.tree import DecisionTreeClassifier

from model_algorithms.model import Model


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

    def get_eval_metrics(self):
        """
        Provides metrics specific to the model for later visualisation (optional).

        :return: list of metrics. Format for metric:
            metrics = [{'name': plot_name, 'x': (x_label, x_values), 'y': (y_label, y_values)}, ...]
        """
        return []
