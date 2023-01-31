import numpy as np

from model_algorithms.model import Model


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
