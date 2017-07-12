"""
This module trains a tertiary classifier
Splits a dataset into training and testing
Inputs are:
    subject, body, event_type, predicted_primary_class, predicted_primary_score
Ouputs:
    DataFrame with the following columns:
        subject, body, event_type, predicted_primary_class,
        predicted_primary_score, predicted_secondary_class,
        predicted_secondary_score
"""

# pylint: disable=invalid-name

import os
import time
import logging
import argparse

# helper modules

import context
from helpers.ml_utils import (precision_recall_matrix, _check_col_dimension, _check_row_dimension,
                              get_classifier_results, grid_search, get_ensemble_prediction,
                              save_feature_pipeline, save_classifier, save_dataset,
                              _check_prediction_dimensions, CLASSIFIER_PIPELINE)
from helpers.nlp_helper import CleanTextTransformer, tokenize_text

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from scipy.sparse import hstack

filepath = os.path.dirname(__file__)
CLASSIFIER_DIR = os.path.join(filepath, '../pickles/classifiers/')
FEATURE_DIR = os.path.join(filepath, '../pickles/feature_pipelines/')
OUTPUT_DIR = os.path.join(filepath, '../pickles/output/')
HIERARCHY_DIR = os.path.join(filepath, '../pickles/hierarchy.csv')
LOGPATH = os.path.join(filepath, '../log/event_classifier_log')

HIERARCHY = pd.read_csv(HIERARCHY_DIR)

logging.basicConfig(filename=LOGPATH,
                    filemode='a',
                    format='%(asctime)s -  %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='This is the event classifier program.')

parser.add_argument('--id', help='Load existing classifier. Usage: --id 485.pkl', required=True)

TERTIARY_FEATURE_PIPELINE = dict({
    'subject_pipe': Pipeline([
        ('cleanText', CleanTextTransformer()),
        ('vectorizer', CountVectorizer(tokenizer=tokenize_text, ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer())
        ]),
    'text_pipe': Pipeline([
        ('cleanText', CleanTextTransformer()),
        ('vectorizer', CountVectorizer(tokenizer=tokenize_text, ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer())
        ]),
    'event_type_pipe': Pipeline([('vectorizer', CountVectorizer())]),
    'predicted_primary': Pipeline([('vectorizer', CountVectorizer())]),
    'predicted_secondary': Pipeline([('vectorizer', CountVectorizer())])
})


def get_tertiary_training_features(X, y, feature_pipeline):
    """
    Combines training features
    Using fit_transform on the pipeline object for each feature
    Performs sparse matrix concatenation
    """

    logger.info("Beginning pipeline fit_transform to training data...")

    subject_matrix = feature_pipeline['subject_pipe'].fit_transform(X.subject, y)
    text_matrix = feature_pipeline['text_pipe'].fit_transform(X.text, y)
    event_type_matrix = feature_pipeline['event_type_pipe'].fit_transform(X.event_type, y)
    predicted_primary_matrix = feature_pipeline['predicted_primary'].fit_transform(
        X.final_primary, y)
    predicted_secondary_matrix = feature_pipeline['predicted_secondary'].fit_transform(
        X.final_secondary, y)

    X = hstack([subject_matrix, text_matrix, event_type_matrix, predicted_primary_matrix,
                predicted_secondary_matrix])

    logger.info("Completed fit_transform")

    _check_col_dimension(X, subject_matrix, text_matrix, event_type_matrix,
                         predicted_primary_matrix, predicted_secondary_matrix)
    _check_row_dimension(X, subject_matrix, text_matrix, event_type_matrix,
                         predicted_primary_matrix, predicted_secondary_matrix)

    logger.info("Training set dimension: %s", X.shape)

    return X, feature_pipeline


def get_tertiary_testing_features(X, feature_pipeline):
    """
    Combines training features
    Key difference between this and training_features
    is that pipeline is transforming x_test not fit_transforming
    """

    logger.info("Beginning transform of test set...")

    subject_matrix = feature_pipeline['subject_pipe'].transform(X.subject)
    text_matrix = feature_pipeline['text_pipe'].transform(X.text)
    event_type_matrix = feature_pipeline['event_type_pipe'].transform(X.event_type)
    predicted_primary_matrix = feature_pipeline['predicted_primary'].transform(
        X.final_primary)
    predicted_secondary_matrix = feature_pipeline['predicted_secondary'].transform(
        X.final_secondary)

    X = hstack([subject_matrix, text_matrix, event_type_matrix, predicted_primary_matrix,
                predicted_secondary_matrix])

    logger.info("Completed transform of test set.")

    _check_col_dimension(X, subject_matrix, text_matrix, event_type_matrix,
                         predicted_primary_matrix, predicted_secondary_matrix)
    _check_row_dimension(X, subject_matrix, text_matrix, event_type_matrix,
                         predicted_primary_matrix, predicted_secondary_matrix)

    logger.info("Testing set dimension: %s", X.shape)

    return X


def _decorate_with_tertiary(X, *args):
    """
    Adds two columns to x_test
        predicted primary class
        predicted primary score
    Used as input to secondary classifier
    """

    # uses output of primary model
    X['true_tertiary_class'] = args[0]
    X['predicted_tertiary_class'] = args[1]
    X['predicted_tertiary_score'] = args[2]

    return X


def _enforce_tertiary_hierarchy(row):
    """
    Enforce Tertiary hierarchy according to set of rules
    """

    secondary, s_score, tertiary, t_score = row['final_secondary'], row['predicted_secondary_score'], \
        row['predicted_tertiary_class'], row['predicted_tertiary_score']

    expected_secondary = np.unique(HIERARCHY[HIERARCHY.t_class == tertiary].s_class)

    if secondary in expected_secondary:
        # tertiary hierarchy enforced
        return secondary, tertiary

    else:

        if s_score > t_score:  # primary wins
            return secondary, 'unknown'
        else:
            if len(expected_secondary) == 2:
                return 'unknown', tertiary
            return expected_secondary[0], tertiary


def add_final_tertiary_classes(X):
    """
    Enforce the hierarchy and add final classification
    """

    print("Adding final classifications to dataframe...")
    finals = X.apply(_enforce_tertiary_hierarchy, axis=1)
    finals = np.array([np.array([i[0], i[1]]) for i in finals])

    X.loc[:, 'final_secondary'] = finals[:, 0]
    X.loc[:, 'final_tertiary'] = finals[:, 1]

    return X


def main():
    """
    Main function that kicks off the program
    """

    start_time = time.time()
    logger.info("Training new tertiary model...")

    args = parser.parse_args()

    id_num = args.id

    df = pd.read_pickle(OUTPUT_DIR + "_secondary_dataset_" + str(id_num) + ".pkl")

    y = df['true_tertiary_class']
    X = df[['subject', 'text', 'event_type', 'final_primary', 'final_secondary',
            'predicted_primary_score', 'predicted_secondary_score']]

    logger.info("Size of X input: %s", X.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    x_train_matrix, feature_pipeline = get_tertiary_training_features(
        x_train, y_train, TERTIARY_FEATURE_PIPELINE)
    x_test_matrix = get_tertiary_testing_features(
        x_test, feature_pipeline)

    save_feature_pipeline(feature_pipeline, 'tertiary', id_num)

    clf_list, clf_names = grid_search(x_train_matrix, y_train, CLASSIFIER_PIPELINE)

    results, classes = get_classifier_results(clf_list, clf_names, x_test_matrix, y_test)
    save_classifier(clf_list, 'tertiary', id_num)

    y_pred, y_score = get_ensemble_prediction(results, classes)
    _check_prediction_dimensions(y_test, y_pred, y_score)

    x_test = _decorate_with_tertiary(x_test, y_test, y_pred, y_score)
    x_test = add_final_tertiary_classes(x_test)

    # Calculate the accuracy of the model
    logger.info("------------------------------------------------")
    logger.info("Overall Accuracy Tertiary: %s",
                accuracy_score(x_test.true_tertiary_class, x_test.final_tertiary))
    print(precision_recall_matrix(
        x_test.true_tertiary_class, x_test.final_tertiary, classes))

    logger.info("Storing Tertiary model output and saving...")
    # saving true tertiary classes
    save_dataset(x_test, 'tertiary', id_num)

    logger.info('Completed training of tertiary model %s', id_num)
    logger.debug("Elapsed time: %s", time.time() - start_time)


if __name__ == '__main__':
    main()
