import numpy as np
import pandas as pd

class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        X = X.astype('float')
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)]
            for i in separated])
        return self

    def _prob(self, x, mean, std):
        exponent = np.exp(- ((x - mean)**2 / (2 * std**2)))
        return np.log(exponent / (np.sqrt(2 * np.pi) * std))

    def predict(self, X):
        X = X.astype('float')
        k =  [[sum(self._prob(i, *s) for s, i in zip(summaries, x)) for summaries in self.model] for x in X]
        return np.argmax(k, axis=1)
    
    def score(self, X, y):
        return sum(self.predict(X) == y) * 1.0 / len(y)
