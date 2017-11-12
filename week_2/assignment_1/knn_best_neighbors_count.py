"""
coursera.org: introduction to machine learning
week 2, assignment 1: solution for finding the best number of neighbors
materials:
- dataset of wines
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
"""
import pandas
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def best_neighbors_count(begin, end, features, target, kfold):
    """ This function finds the best count of neigbors for dataset """
    k = -1
    accuracy = -1
    for i in xrange(begin, end + 1):
        neigh = KNeighborsClassifier(n_neighbors=i)
        mean = cross_val_score(neigh, features, target, cv=kfold).mean()
        if mean > accuracy:
            accuracy = mean
            k = i

    return (k, accuracy)


def main():
    """ general main function to load, prepare data and output the results """
    data_frame = pandas.read_csv('wine.data', header=None)
    features = data_frame.iloc[:, 1:]
    target = data_frame.iloc[:, :1].values.ravel()

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    print "Q 1-2: " + str(best_neighbors_count(1, 50, features,
                                               target, kfold))
    print "Q 3-4: " + str(best_neighbors_count(1, 50, scale(features),
                                               target, kfold))


if __name__ == "__main__":
    main()
