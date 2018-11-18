import numpy as np
import scipy.optimize as opt
from scipy.optimize import fmin_bfgs
import pandas as pd
import sys

def sigmoid(X):
    return 1.0/(1 + np.exp(-X))


class LogisticRegression:
    def __init__(self):
        pass

    def fit(self,X,y):
        self.X = np.hstack((np.ones((X.shape[0],1)),X))
        self.X = self.X.astype('float')
        self.y = y
        self.nX = X.shape[0]
        self.nEta = X.shape[1]+1
        self.eta = fmin_bfgs(self.costFunction,np.zeros(self.nEta),fprime=self.gradient)

    def score(self,X,y):
        return 1.0 * (self.predict(X) == y).sum() / len(y)

    def predict(self,X):
        X = np.hstack((np.ones((X.shape[0],1)),X))
        X = X.astype('float')
        y = self.sigmoid(np.dot(X,self.eta))
        y = (y>=0.5)
        return y

    def sigmoid(self,z):
        return 1.0 / ( 1.0+np.exp(-z) )

    def costFunction(self,eta):
        prob = self.sigmoid(np.dot(self.X,eta))
        prob_ = 1.0-prob
        np.place(prob,prob==0.0,sys.float_info.min)
        np.place(prob_,prob_==0.0,sys.float_info.min)
        self.cost = np.dot(self.y,np.log(prob))
        self.cost += np.dot((1.0-self.y),np.log(prob_))
        return -self.cost

    def gradient(self,eta):
        grad = np.zeros(self.nEta)
        for i in range(grad.shape[0]):
            grad[i] = ( (self.sigmoid(np.dot(self.X,eta))-self.y)*self.X[:,i] ).sum()
        return grad
