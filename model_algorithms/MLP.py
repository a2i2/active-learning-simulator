from sklearn.neural_network import MLPClassifier

from model_algorithms.model import Model


class MLP(Model):
    def __init__(self):
        self.model = MLPClassifier()

    def train(self, X, Y):
        self.model.fit(X, Y)
        return

    def test(self, X, Y):
        y_prob = self.model.predict_proba(X)
        return y_prob

    def predict(self, X, Y):
        y_pred = self.model.predict(X)
        return y_pred

    def reset(self):
        self.model = MLPClassifier()
        return

    def get_eval_metrics(self):
        return
