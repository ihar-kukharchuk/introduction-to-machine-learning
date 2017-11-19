"""
coursera.org: introduction to machine learning
week 3, assignment 2: using SVM and TD-IDF determine the most
                      weighable words in atheism or space topics
"""
import numpy
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer


def read_data():
    """ this function reads data file and split it to features and target """
    newsgroups = datasets.fetch_20newsgroups(
        subset='all',
        categories=['alt.atheism', 'sci.space']
    )
    return newsgroups.data, newsgroups.target


def get_feature_to_weight_map(features, target, c_constant):
    """ this function zips feature indexes with theirs weights """
    clf = svm.SVC(kernel='linear', random_state=241, C=c_constant)
    clf.fit(features, target)

    return zip(clf.coef_.indices, abs(clf.coef_.data))


def main():
    """ general main function to load, prepare data and output the results """
    features, target = read_data()

    # convert 'text' to real representation
    vectorizer = TfidfVectorizer()
    vectorized_features = vectorizer.fit_transform(features)

    # calculates the best 'C' for SVM method
    grid = {'C': numpy.power(10.0, numpy.arange(-5, 6))}
    crossval = KFold(n_splits=5, shuffle=True, random_state=241)
    clf = svm.SVC(kernel='linear', random_state=241)
    gridsearch = GridSearchCV(clf, grid, scoring='accuracy', cv=crossval,
                              return_train_score=True, n_jobs=8)
    gridsearch.fit(vectorized_features, target)
    best_c = gridsearch.cv_results_['params'][gridsearch.best_index_]['C']

    # find the most weighable words in provided texts
    ftw = get_feature_to_weight_map(vectorized_features, target, best_c)
    ftw.sort(key=lambda pair: -pair[1])
    words = [vectorizer.get_feature_names()[pair[0]] for pair in ftw[0:10]]

    print "Q 1: " + ' '.join(sorted(words))


if __name__ == "__main__":
    main()
