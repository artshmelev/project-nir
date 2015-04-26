import sys
import time
import itertools
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcess


class COGP(object):

    def __init__(self, func, initial=None, minimize=True, storeAllEvaluations=True,
                 storeAllEvaluated=True, maxEvaluations=10, grid=None):
        self.func = func
        self.coef = 1. if minimize else -1.

        if not initial and not grid:
            raise Exception('Error: you should set initial or grid parameter')

        if initial:
            self.X = np.array(initial)
            self.y = np.array([self.coef*func(el) for el in initial])
            LEFT = np.min(initial)
            RIGHT = np.max(initial)
            self.x = [i for i in itertools.product(np.arange(LEFT, RIGHT, 0.1),
                                                   repeat=len(initial[0]))]
        else:
            lst = [grid[0], grid[-1]]
            self.X = np.array(lst)
            self.y = np.array([self.coef*func(el) for el in lst])
            self.x = grid

        self.fmin = np.min(self.y)
        self.argmin = self.X[np.argmin(self.y)]
        self.gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4,
                                  thetaU=1e-1)
        self.storeAllEvaluations = storeAllEvaluations
        self.storeAllEvaluated = storeAllEvaluated
        if storeAllEvaluations:
            self._allEvaluations = []
        if storeAllEvaluated:
            self._allEvaluated = []
        self.max_evaluations = maxEvaluations
        self.time = 0

    def learn(self):
        wall_time = time.time()
        for _ in xrange(self.max_evaluations):
            try:
                self.gp.fit(self.X, self.y)
            except:
                new_x = None
                for x in self.x:
                    if not any(np.equal(self.X, x).all(1)):
                        new_x = x
                        break
                if not new_x:
                    break

                self.X[-1] = np.array(new_x)
                f = self.coef*self.func(new_x)
                if self.storeAllEvaluations:
                    self._allEvaluations[-1] = f
                if self.storeAllEvaluated:
                    self._allEvaluated[-1] = new_x
                self.y[-1] = np.array(f)
                if f < self.fmin:
                    self.fmin = f
                    self.argmin = new_x

                self.gp.fit(self.X, self.y)

            y_pred, MSE = self.gp.predict(self.x, eval_MSE=True)

            s = (self.fmin-y_pred) / np.sqrt(MSE)
            argm = np.argmax(MSE * (s * norm.cdf(s) + norm.pdf(s)))

            self.X = np.vstack([self.X, self.x[argm]])
            f = self.coef*self.func(self.x[argm])
            if self.storeAllEvaluations:
                self._allEvaluations.append(f)
            if self.storeAllEvaluated:
                self._allEvaluated.append(self.x[argm])
            self.y = np.hstack([self.y, f])
            if f < self.fmin:
                self.fmin = f
                self.argmin = self.x[argm]

        self.time = time.time() - wall_time
        return (np.array(self.argmin), self.coef*self.fmin)
