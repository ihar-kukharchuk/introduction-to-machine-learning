"""
coursera.org: introduction to machine learning
week 2, assignment 2: solution for finding the best 'p' parameter
                      for minkowski metric to build regression
materials:
- dataset features explanation
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
"""
import numpy
import sklearn
import sklearn.datasets
import sklearn.preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def minkowski_kernel(features, target, p_arg):
    """ this function calculates cross_val_score for provided parameters """
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance',
                                metric='minkowski', p=p_arg)
    return cross_val_score(neigh, features, target, cv=kfold,
                           scoring='neg_mean_squared_error')


def best_minkowski_p(start, stop, num, features, target):
    """ This function finds the best 'p' parameter for minlowski metric """
    p_best = start
    min_mistake = 0

    for i in numpy.linspace(start, stop, num):
        res = max(minkowski_kernel(features, target, i))
        if -res < min_mistake or i == start:
            min_mistake = -res
            p_best = i

    return (p_best, min_mistake)


def main():
    """ general main function to load, prepare data and output the results """
    dataset = sklearn.datasets.load_boston()
    dataset.data = sklearn.preprocessing.scale(dataset.data)

    print "Q 1: " + str(best_minkowski_p(1, 10, 200,
                                         dataset.data, dataset.target))


if __name__ == "__main__":
    main()
