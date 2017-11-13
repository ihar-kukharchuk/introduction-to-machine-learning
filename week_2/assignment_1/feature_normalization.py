"""
coursera.org: introduction to machine learning
week 2, assignment 3: feature normalization experiments during
                      perceptron learning procedure
"""
import pandas
import sklearn
import sklearn.preprocessing
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler


def read_data(filename):
    """ this function reads data file and split it to features and target """
    train_set = pandas.read_csv(filename, header=None)
    return (train_set.iloc[:, 1:], train_set.iloc[:, :1].values.ravel())


def get_accuracy(train_features, train_target, test_features, test_target):
    """ this function calculates accuracy of learned perceptron """
    clf = Perceptron(random_state=241, max_iter=1000)
    clf.fit(train_features, train_target)
    return sklearn.metrics.accuracy_score(test_target,
                                          clf.predict(test_features))


def main():
    """ general main function to load, prepare data and output the results """
    train_features, train_target = read_data('perceptron-train.csv')
    test_features, test_target = read_data('perceptron-test.csv')

    accuracy = get_accuracy(train_features, train_target,
                            test_features, test_target)

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    accuracy_scaled = get_accuracy(train_features_scaled, train_target,
                                   test_features_scaled, test_target)

    print "Q 1: " + str(accuracy_scaled - accuracy)


if __name__ == "__main__":
    main()
