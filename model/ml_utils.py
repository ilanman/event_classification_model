"""
This file contains helper functions for performing common ML tasks
"""

import os
import pandas as pd
import _pickle as cPickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def read_data_from_json(path):
    """
    Input: path to a directory of files stored as .json

    Returns: dataframe with each JSON file stored per row. The column headers
    of the dataframe are the JSON keys
    """

    json_files = [os.path.join(path, f) for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f))
                  and f.endswith('.json')]
    data_frame = pd.DataFrame()
    counter = 0

    for file_name in json_files:
        data = pd.read_json(file_name, typ='series')
        data_frame = data_frame.append(data, ignore_index=True)
        counter += 1

        if counter % 5000 == 0:
            print('Read {} files'.format(counter))

    return data_frame


def plot_roc_curve(y_test, y_probs, classifier):
    """
    Makes an ROC curve
    """

    y_prob_spam = y_probs.T[1]
    fpr, tpr, _ = roc_curve(y_test, y_prob_spam)
    plt.figure(figsize=(10, 8))
    plt.title("ROC Curve for Spam/Ham Classifier", size=16)
    plt.plot(fpr, tpr, label=classifier)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.ylabel('true positive rate', size=14)
    plt.xlabel('false positive rate', size=14)
    plt.legend(loc='best', fontsize=14)
    print("AUC =", auc(fpr, tpr))
    plt.show()


def get_preds(clf, X_test):
    """
    Uses the best estimate from the GridSearch and makes predictions and scores
    """

    y_probs = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)

    return y_probs, y_pred


def pickle_classifier(clf, path):
    """
    Save classifier to disk
    """

    # save the classifier
    print("Saving classifier: {}...".format(path))
    with open(path + ".pkl", 'wb') as f:
        cPickle.dump(clf, f)


def load_classifier(path):
    """
    Load pickled classifier given its path
    """

    print("Loading classifier: {}...".format(path))
    with open(path + ".pkl", 'rb') as f:
        clf = cPickle.load(f)

    return clf
