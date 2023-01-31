from sklearn.linear_model import LogisticRegression

from model_algorithms.model import Model


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
