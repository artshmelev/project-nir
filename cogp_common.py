import sys
import itertools
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcess

np.random.seed(1)
LEFT = -10.
RIGHT = 10.
NUMP = 400
STEP = (RIGHT-LEFT) / NUMP


class COGP(object):

    def __init__(self, func, initial, minimize=True,
                 storeAllEvaluations=True, storeAllEvaluated=True,
                 maxEvaluations=10):
        self.func = func
        self.X = np.array(initial)
        self.y = np.array([func(el) for el in initial])
        self.num_params = len(initial[0])
        self.x = [i for i in itertools.product(np.arange(LEFT, RIGHT, STEP),
                                               repeat=self.num_params)]
        self.fmin = np.min(self.y)
        self.argmin = self.X[np.argmin(self.y)]
        self.gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4,
                                  thetaU=1e-1, random_start=100)
        self.max_evaluations = maxEvaluations

    def learn(self):
        for step in xrange(self.max_evaluations):
            try:
                self.gp.fit(self.X, self.y)
            except:
                break
            y_pred, MSE = self.gp.predict(self.x, eval_MSE=True)
            sigma = np.sqrt(MSE)

            s = (self.fmin-y_pred) / sigma
            delta = sigma * (s * norm.cdf(s) + norm.pdf(s))

            argm = np.argmax(sigma * delta)
            print self.x[argm]

            self.X = np.vstack([self.X, self.x[argm]])
            f = self.func(self.x[argm])
            self.y = np.hstack([self.y, f])
            if f < self.fmin:
                self.fmin = f
                self.argmin = self.x[argm]
        return (self.argmin, self.fmin)


def f(x):
    return x[0] * np.sin(x[0])


def f1(x):
    return (x[0]-0.2)**2 + (x[1])**2 + (x[2])**2 + (x[3])**2


def branin(x):
    a = 1
    b = 5 / (4*np.pi*np.pi)
    c = 5 / np.pi
    d = 6
    e = 10
    f = 1 / (8*np.pi)
    return a*(x[1] - b*x[0]*x[0] + c*x[0] - d)**2 + e*(1-f)*np.cos(x[0]) + e


def colville(x):
    return 100*(x[0]**2-x[1])**2 + (x[0]-1)**2 + (x[2]-1)**2 + \
        90*(x[2]**2-x[3])**2 + 10.1*((x[1]-1)**2+(x[3]-1)**2) + \
        19.8*(x[1]-1)*(x[3]-1)


if __name__ == '__main__':
    # l = COGP(f, [[0., 0.], [9., 9.]], maxEvaluations=int(sys.argv[1]))
    l = COGP(branin, [[-10., -10.], [-10., 10.], [10., -10.], [10., 10.],
                      [0., 0.]],
             maxEvaluations=int(sys.argv[1]))
    """
    l = COGP(colville, [[0.,0.,0.,0.], [2.,2.,2.,2.],
                        [0.,0.,2.,2.], [2.,2.,0.,0.],
                        [0.,2.,0.,2.], [2.,0.,2.,0.],
                        [0.,2.,2.,0.], [2.,0.,0.,2.],
                        [1.1, 1., 1., 1.]], maxEvaluations=int(sys.argv[1]))
    """
    # l = COGP(f1, [[-5.,-5.,-5.,-5.], [5.,5.,5.,5.]],
    #          maxEvaluations=int(sys.argv[1]))

    out = l.learn()
    print out
