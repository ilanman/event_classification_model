"""
This module contains helper functions shared by the event classifier models
primary, secondary, tertiary, both training full models and scoring individual event
"""

# pylint: disable=invalid-name

import os
import logging
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
import numpy as np
import _pickle as cPickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import BaseEstimator, Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

filepath = os.path.dirname(__file__)
CLASSIFIER_DIR = os.path.join(filepath, '../pickles/classifiers/')
LOGPATH = os.path.join(filepath, '../log/event_classifier_log')
FEATURE_DIR = os.path.join(filepath, '../pickles/feature_pipelines/')
OUTPUT_DIR = os.path.join(filepath, '../pickles/output/')
logging.basicConfig(filename=LOGPATH,
                    filemode='a',
                    format='%(asctime)s -  %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


# parameters for gridsearch
# using SVM currently
CLASSIFIER_PIPELINE = dict({
    'LogisticRegression': {
        'classifier': Pipeline([
            ("clf", LogisticRegression()),
        ]),
        'params': {
            'clf__C': [1],
            }
        },

    'SVM': {
        'classifier': Pipeline([
            ("clf", SVC(probability=True, class_weight='balanced', kernel='linear')),
        ]),
        'params': {
            'clf__C': [1, 3],
            }
        },
})


def read_data_from_json(path):
    """
    Input: path to a directory of files stored as .json

    Returns: dataframe with each JSON file stored per row. The column headers
    of the dataframe are the JSON keys
    """

    json_files = [os.path.join(path, f) for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f)) and
                  f.endswith('.json')]
    data_frame = pd.DataFrame()
    counter = 0

    for file_name in json_files:
        data = pd.read_json(file_name, typ='series')
        data_frame = data_frame.append(data, ignore_index=True)
        counter += 1

        if counter % 5000 == 0:
            print('Read {} files'.format(counter))

    return data_frame


def plot_roc_curve(y_test, y_probs, classifier_name):
    """
    Makes an ROC curve
    """

    y_prob_spam = y_probs.T[1]
    fpr, tpr, _ = roc_curve(y_test, y_prob_spam)
    plt.figure(figsize=(10, 8))
    plt.title("ROC Curve for {} Classifier".format(classifier_name), size=16)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.ylabel('true positive rate', size=14)
    plt.xlabel('false positive rate', size=14)
    plt.legend(loc='best', fontsize=14)
    print("AUC =", auc(fpr, tpr))
    plt.show()


def save_feature_pipeline(feature_pipeline, level, id_num):
    """
    Save training features
    Need to create an id_num to know which classifier to load
    Loaded classifier must be the same as the feature id number
    """

    if level == 'primary':
        id_num = np.random.randint(1000)

    logger.info("Saving %s fitted feature pipeline id %s...", level, id_num)
    pickle_classifier(feature_pipeline, FEATURE_DIR + level + "_" + str(id_num))

    return id_num


def get_preds(clf, X_test):
    """
    Uses the best estimate from the GridSearch and makes predictions and scores
    """

    try:
        y_probs = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)

    except AttributeError as e:
        print(e)
        print("Make sure model has predict_proba functionality")

    return y_probs, y_pred


def save_classifier(clf, level, id_num):
    """
    Save trained classifier
    """

    logger.info("Saving trained classifier id %s...", id_num)
    pickle_classifier(clf, CLASSIFIER_DIR + level + "_" + str(id_num))


def save_dataset(X, level, id_num):
    """
    Saves the output to a file
    """

    logger.info("Saving %s output for id %s...", level, id_num)
    pickle_classifier(X, OUTPUT_DIR + "_" + level + "_dataset_" + str(id_num))


def pickle_classifier(clf, path):
    """
    Save classifier to disk
    """

    # save the classifier
    print("Saving classifier: {}...".format(path))
    with open(path + ".pkl", 'wb') as pkl:
        cPickle.dump(clf, pkl)


def load_classifier(path):
    """
    Load pickled classifier given its path
    """

    print("Loading classifier: {}...".format(path))
    with open(path + ".pkl", 'rb') as pkl:
        clf = cPickle.load(pkl)

    return clf


def precision_recall_matrix(y_test, y_pred, classes):
    """
    Use results of classifiers to create a precision and recall matrix
    """

    #  Create the precision and recall matrix
    result_prec_recall = precision_recall_fscore_support(y_test, y_pred, labels=classes)
    result = pd.DataFrame()
    result['classification'] = classes
    result['precision'] = result_prec_recall[0]
    result['recall'] = result_prec_recall[1]

    # Display the precision and recall table
    return result


def get_classifier_results(clf_list, clf_names, X, y=None):
    """
    Input:
        clf_list: list of classifiers
        clf_names: name of each classifier
        X: set to score against, usually a testing set
        y: target variable (optional)
    Stores results of each classifier in a dictionary
    First element is the predicted class, clf.predict()
    Second element is the probability, clf.predict_proba()
    Ensure that each classifier passed has a predict_proba() method

    Prints the accuracy acheived for each classifier
    Returns a dictionary of results and the class labels
    """

    print("Getting classifier results...")
    results = {}
    classes = clf_list[0].classes_

    for (name, clf) in zip(clf_names, clf_list):

        results[name] = [clf.predict(X), clf.predict_proba(X)]

        # if classifier is being used on a testing set, then log its accuracy
        if y is not None:

            logger.info("------------------------------------------------")
            logger.info("%s Accuracy: %s", name, accuracy_score(y, results[name][0]))

    return results, classes


def grid_search(X, y, gridsearch_pipeline):
    """
    Perform a Grid Search over the space of classifiers and their associated
    parameter space
    Inputs: X and y training sets
    Output: A list of the best classifiers from each classifier category
    """

    logger.info("starting Gridsearch...")

    best_classifiers = []
    names = []

    for v in gridsearch_pipeline.items():
        gs = GridSearchCV(v[1]['classifier'], v[1]['params'], verbose=2, cv=3, n_jobs=4)
        gs = gs.fit(X, y)
        names.append(v[0])
        logger.info("%s finished", v[0])
        logger.info("Best scoring classifier: %s", gs.best_score_)
        best_classifiers.append(gs.best_estimator_)

    return best_classifiers, names


class ExtractFeature(BaseEstimator, TransformerMixin):
    """
    Extract the correct feature by column name from a pandas dataframe
    Used to perform pipeline operations on
    """

    def __init__(self, key):
        """
        Returns the key
        """
        self.key = key

    def fit(self, x, y=None):
        """
        Scikit requires a fit function
        """

        return self

    def transform(self, df):
        """
        Returns subset of dataframe where column == key
        """
        try:
            return df[self.key]
        except KeyError as err:
            print(err)
            print("doesn't exist in dataframe.")
            return


def _most_common(lst):
    """
    Using Counter find most common element in list
    Possible results:
        mc = [(val1,3)]
        mc = [(val1,2),(val2,1)]
        mc = [(val1,1),(val2,1),(val3,1)]
        ...
    No majority exists only when there is a tie, otherwise the first value of
    the list is the most common (because Counter sorts automatically in
    descending order by value)
    """

    # check if top two most common predictions are the same
    mc = Counter(lst).most_common(2)

    if len(mc) > 1 and mc[0][1] == mc[1][1]:
        return "no_majority"

    return mc[0][0]


def get_ensemble_prediction(results, classes):
    """
    if majority of classifiers choose same category, that's the winner.
    if majority does not exist, then select class with highest probability
    """

    print("Getting ensemble predictions...")
    num_clfs = len(results)

    # combine all classifier predictions and probabilities
    all_preds = np.array([v[0] for k, v in results.items()]).T
    all_probs = np.sum(np.array([v[1] for k, v in results.items()]), axis=0)
    all_probs_normalize = all_probs/num_clfs

    # the function most_common returns majority class or "no_majority"
    majority = np.array(list(map(_most_common, all_preds)))
    no_majority_index = np.where(majority == 'no_majority')

    # for those where a majority doesn't exist, sum the probabilities for each
    # class
    no_majority_sum = all_probs_normalize[no_majority_index]

    # ensure no new probabilities were added that shouldn't be
    assert np.allclose(np.sum(no_majority_sum), len(no_majority_index[0])), (
        "probability sum is greater than expected for no_majority")

    # replace the "no_majority" samples with the class that resulted in the
    # largest probability
    majority[no_majority_index] = classes[np.argmax(no_majority_sum, axis=1)]

    return majority, np.max(all_probs_normalize, axis=1)


def _check_col_dimension(X, *args):
    """
    Ensure number of rows matches
    """

    col_sum = 0
    for mat in args:
        col_sum += mat.shape[1]

    assert (X.shape[1] == col_sum), (
        "Number of training features doesn't match sum of component features")


def _check_row_dimension(X, *args):
    """
    Ensure number of rows matches
    """

    for arg in args:
        assert (X.shape[0] == arg.shape[0]), (
            "Number of training samples doesn't match sum of component samples")


def _check_prediction_dimensions(y_test, y_pred, y_score):
    """
    Ensure that prediction dimensions are correct
    """

    assert y_pred.shape[0] == y_test.shape[0], (
        "Ensure class prediction vector is same length as test set")
    assert y_score.shape[0] == y_test.shape[0], (
        "Ensure score prediction vector is same length as test set")


def load_classifier_list(id_num, level):
    """
    Load a classifier
    Classifier is stored as list object
    Returns list of classifiers and their names
    """

    print("Loading classifier list...")
    # clean up input: remove "../pickle/classifier/" and ".pkl"

    clf_list, clf_names = [], []

    # this is a list of classifiers
    loaded_clf = load_classifier(CLASSIFIER_DIR + level + "_" + str(id_num))

    for classifier in loaded_clf:
        clf_list.append(classifier)

        # get name via class structure
        clf_class = str(classifier.named_steps['clf'].__class__)

        # some basic cleaning of class name
        clf_name_indx = clf_class.find('.')
        clf_name = clf_class[clf_name_indx+1:-2]
        clf_names.append(clf_name)

    return clf_list, clf_names


def create_date_range(start, end):
    """
    Creates a list of dates of format, ["2017-06-01", "2017-06-02", ...]
    From start_date to end_date
    """

    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, "%Y-%m-%d")
    numdays = (end_date - start_date).days
    date_list = [start_date + timedelta(days=x) for x in range(0, numdays)]

    # converts to "2017-06-01" format
    formatted_date_list = [d.date().isoformat() for d in date_list]

    return formatted_date_list
