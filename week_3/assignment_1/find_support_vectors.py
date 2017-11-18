"""
coursera.org: introduction to machine learning
week 3, assignment 1: find support vectors in SVM method
"""
import pandas
from sklearn import svm


def read_data(filename):
    """ this function reads data file and split it to features and target """
    train_set = pandas.read_csv(filename, header=None)
    return (train_set.iloc[:, 1:], train_set.iloc[:, :1].values.ravel())


def get_support_vectors_indexes(features, target, c_constant):
    """ this function calculates support vectors indexes for data """
    clf = svm.SVC(C=c_constant, kernel='linear', random_state=241)
    clf.fit(features, target)
    return clf.support_


def main():
    """ general main function to load, prepare data and output the results """
    features, target = read_data('svm-data.csv')

    indexes = get_support_vectors_indexes(features, target, 100000)

    print "Q 1: " + str(indexes + 1)


if __name__ == "__main__":
    main()
