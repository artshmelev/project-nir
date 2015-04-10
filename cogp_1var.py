import sys
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl

np.random.seed(1)
LEFT = -3
RIGHT = 8
NUMP = 1000

class COGP(object):
    def __init__(self, func, initial, minimize=True,
                 storeAllEvaluations=True, storeAllEvaluated=True,
                 maxEvaluations=10):
        self.func = func
        self.lst = initial
        self.gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4,
                                  thetaU=1e-1, random_start=100)
        self.max_evaluations = maxEvaluations

    def learn(self):
        x = np.atleast_2d(np.linspace(LEFT, RIGHT, NUMP)).T

        for step in xrange(self.max_evaluations):
            X = np.atleast_2d(self.lst).T
            y = self.func(X).ravel()
            f_min = min(y)
            try:
                self.gp.fit(X, y)
            except:
                break
            y_pred, MSE = self.gp.predict(x, eval_MSE=True)
            sigma = np.sqrt(MSE)

            s = (f_min-y_pred) / sigma
            delta = sigma * (s * norm.cdf(s) + norm.pdf(s))

            x_prod = float(np.argmax(sigma * delta)) / NUMP * (RIGHT-LEFT) + LEFT
            self.lst.append(x_prod)

        return (self.lst[-1], self.func(self.lst[-1]))
            

def f(x):
    return x * np.sin(x)

def f1(x):
    return (x-2) ** 2

if __name__ == '__main__':
    if sys.argv[1] == 'compute':
        l = COGP(f, [0., 9.], maxEvaluations=int(sys.argv[2]))
        out = l.learn()
        print out
    elif sys.argv[1] == 'plot':
        lst = [float(LEFT), float(RIGHT)]
        x = np.atleast_2d(np.linspace(LEFT, RIGHT, NUMP)).T
        gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4,
                             thetaU=1e-1, random_start=100)

        while True:
            lst.append(float(input('enter new point=')))
            X = np.atleast_2d(lst).T
            y = f(X).ravel()
            f_min = min(y)

            gp.fit(X, y)

            y_pred, MSE = gp.predict(x, eval_MSE=True)
            sigma = np.sqrt(MSE)
            x_sigma = float(np.argmax(sigma)) / NUMP * (RIGHT-LEFT) + LEFT

            s = (f_min-y_pred) / sigma
            delta = sigma * (s * norm.cdf(s) + norm.pdf(s))
            x_delta = float(np.argmax(delta)) / NUMP * (RIGHT-LEFT) + LEFT

            x_prod = float(np.argmax(sigma * delta)) / NUMP * (RIGHT-LEFT) + LEFT

            print 'x_sigma=', x_sigma
            print 'x_delta=', x_delta
            print 'use=', x_prod

            fig = pl.figure()
            pl.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
            pl.plot(X, y, 'r.', markersize=10, label=u'Observations')
            pl.plot(x, y_pred, 'b-', label=u'Prediction')
            pl.plot(x, delta, 'g-', label=u'delta')
            pl.plot(x, sigma, 'y-', label=u'sigma')

            pl.xlabel('$x$')
            pl.ylabel('$f(x)$')
            pl.ylim(-7, 10)
            pl.legend(loc='upper left')

            pl.show()

