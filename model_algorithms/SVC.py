import numpy as np
from sklearn.svm import LinearSVC

from model_algorithms.model import Model


class SVC(Model):
    def __init__(self):
        self.model = LinearSVC()

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
