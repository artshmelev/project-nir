import numpy as np

from sklearn.base import is_classifier, clone
from sklearn.cross_validation import _fit_and_score
from sklearn.cross_validation import _check_cv as check_cv
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import ParameterGrid
from sklearn.grid_search import _CVScoreTuple
from sklearn.metrics.scorer import check_scoring
from sklearn.svm import SVC
from sklearn.utils.validation import _num_samples, indexable

from cogp_test import COGP


class COGPGridSearch(GridSearchCV):

    def _grid_to_simple_list(self, grid):
        result = []

        first = list(grid)[0]
        keys = []
        for k in first.keys():
            if isinstance(first[k], float) or isinstance(first[k], int):
                keys.append(k)

        for el in grid:
            lst = []
            for k in keys:
                if isinstance(el[k], int):
                    lst.append(int(el[k]))
                else:
                    lst.append(float(el[k]))
            result.append(tuple(lst))

        return result, first

    def _list_to_grid_point(self, lst, grid):
        index = 0
        result = list(grid)[0]
        for k in result.keys():
            if isinstance(result[k], float) or isinstance(result[k], int):
                result[k] = lst[index]
                index += 1

        return result

    def fit(self, X, y=None):
        parameter_iterable = ParameterGrid(self.param_grid)
        param_list, first = self._grid_to_simple_list(parameter_iterable)
        estimator = self.estimator
        cv = self.cv
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        n_samples = _num_samples(X)
        X, y = indexable(X, y)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
        cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

        base_estimator = clone(self.estimator)

        n_folds = len(cv)
        grid_scores = list()

        def func(x):
            parameters = self._list_to_grid_point(x, parameter_iterable)

            n_test_samples = 0
            score = 0
            all_scores = []

            for train, test in cv:
                this_score, this_n_test_samples, _, parameters = \
                        _fit_and_score(clone(base_estimator), X, y, self.scorer_,
                                       train, test, self.verbose, parameters,
                                       self.fit_params, return_parameters=True,
                                       error_score=self.error_score)
                all_scores.append(this_score)
                if self.iid:
                    this_score *= this_n_test_samples
                    n_test_samples += this_n_test_samples
                score += this_score

            if self.iid:
                score /= float(n_test_samples)
            else:
                score /= float(n_folds)

            grid_scores.append(_CVScoreTuple(parameters,
                                             score,
                                             np.array(all_scores)))

            #print 'In func:', x, score
            return score

        max_evals = 17 if getattr(self, 'max_evals', None) == None else self.max_evals
        l = COGP(func, maxEvaluations=max_evals, grid=param_list, minimize=False)
        out = l.learn()
        #print 'Out:', out

        self.grid_scores_ = grid_scores

        best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]
        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score

        if self.refit:
            best_estimator = clone(base_estimator).set_params(**best.parameters)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self

    def fit_with_max_evals(self, X, y, max_evals):
        self.max_evals = max_evals
        self.fit(X, y)


if __name__ == '__main__':
    import sys
    import time

    from sklearn.datasets import load_digits

    if sys.argv[1] == '0':
        algorithm = COGPGridSearch
    else:
        algorithm = GridSearchCV
    digits = load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    y = digits.target

    param_grid = [{'kernel': ['rbf'], 'gamma': [1e-5, 1e-4, 1e-2],
                   'C': [1, 5, 10, 50, 100, 200, 500, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print score

        grid_search = algorithm(SVC(C=1), param_grid=param_grid,
                                scoring='%s_weighted' % score)

        start_time = time.time()
        #grid_search.fit_with_max_evals(X, y, 15)
        grid_search.fit(X, y)
        print 'Time:', time.time() - start_time

        print 'Best params:', grid_search.best_params_
        if sys.argv[1] == '1':
            print grid_search.grid_scores_
