"""
This module trains a secondary classifier
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

logging.basicConfig(filename=LOGPATH,
                    filemode='a',
                    format='%(asctime)s -  %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='This is the event classifier program.')

parser.add_argument('--id', help='Load existing classifier. Usage: --id 485.pkl', required=True)

SECONDARY_FEATURE_PIPELINE = dict({
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
    'predicted_primary': Pipeline([('vectorizer', CountVectorizer())])
})

HIERARCHY = pd.read_csv(HIERARCHY_DIR)


def get_secondary_training_features(X, y, feature_pipeline):
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
        X.predicted_primary_class, y)

    X = hstack([subject_matrix, text_matrix, event_type_matrix, predicted_primary_matrix])

    logger.info("Completed fit_transform")

    _check_row_dimension(
        X, subject_matrix, text_matrix, event_type_matrix, predicted_primary_matrix)
    _check_col_dimension(
        X, subject_matrix, text_matrix, event_type_matrix, predicted_primary_matrix)

    logger.info("Training set dimension: %s", X.shape)

    return X, feature_pipeline


def get_secondary_testing_features(X, feature_pipeline):
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
        X.predicted_primary_class)

    X = hstack([subject_matrix, text_matrix, event_type_matrix, predicted_primary_matrix])

    logger.info("Completed transform of test set.")

    _check_row_dimension(
        X, subject_matrix, text_matrix, event_type_matrix, predicted_primary_matrix)
    _check_col_dimension(
        X, subject_matrix, text_matrix, event_type_matrix, predicted_primary_matrix)

    logger.info("Testing set dimension: %s", X.shape)

    return X


def _decorate_with_secondary(X, *args):
    """
    Adds columns to x_test
    Used as input to tertiary classifier
    """

    # uses output of primary model
    X.loc[:, 'true_secondary_class'] = args[0]
    X.loc[:, 'predicted_secondary_class'] = args[1]
    X.loc[:, 'predicted_secondary_score'] = args[2]
    X.loc[:, 'true_tertiary_class'] = args[3]

    return X


def _enforce_secondary_hierarchy(row):

    primary, secondary, p_score, s_score = row['predicted_primary_class'], \
     row['predicted_secondary_class'], row['predicted_primary_score'], \
     row['predicted_secondary_score']

    try:

        expected_primary = np.unique(HIERARCHY[HIERARCHY.s_class == secondary].p_class)

        if primary in expected_primary:
            # Secondary hierarchy enforced
            return primary, secondary
        else:
            if p_score > s_score:  # primary wins
                return primary, 'unknown'
            return expected_primary[0], secondary

    except Exception as e:
        print(e)


def add_final_secondary_classes(X):
    """
    Enforce the hierarchy and add final classification
    """

    print("Adding final classifications to dataframe...")
    finals = X.apply(_enforce_secondary_hierarchy, axis=1)
    finals = np.array([np.array([i[0], i[1]]) for i in finals])

    X.loc[:, 'final_primary'] = finals[:, 0]
    X.loc[:, 'final_secondary'] = finals[:, 1]

    return X


def main():
    """
    Main function that runs the program
    """

    start_time = time.time()
    logger.info("Training new secondary model...")

    args = parser.parse_args()

    id_num = args.id

    df = pd.read_pickle(OUTPUT_DIR + "_primary_dataset_" + str(id_num) + ".pkl")

    y = df['true_secondary_class']
    X = df[['subject', 'text', 'event_type', 'predicted_primary_class', 'predicted_primary_score']]

    logger.info("Size of X input: %s", X.shape)

    x_train, x_test, y_train, y_test, _, idx2 = train_test_split(X, y, X.index, test_size=0.4)

    # saving for future models
    y_tertiary = df.true_tertiary_class[idx2]

    x_train_matrix, feature_pipeline = get_secondary_training_features(
        x_train, y_train, SECONDARY_FEATURE_PIPELINE)
    x_test_matrix = get_secondary_testing_features(
        x_test, feature_pipeline)

    save_feature_pipeline(feature_pipeline, 'secondary', id_num)

    clf_list, clf_names = grid_search(x_train_matrix, y_train, CLASSIFIER_PIPELINE)

    results, classes = get_classifier_results(clf_list, clf_names, x_test_matrix, y_test)
    save_classifier(clf_list, 'secondary', id_num)

    y_pred, y_score = get_ensemble_prediction(results, classes)
    _check_prediction_dimensions(y_test, y_pred, y_score)

    x_test = _decorate_with_secondary(x_test, y_test, y_pred, y_score, y_tertiary)
    x_test = add_final_secondary_classes(x_test)

    # Calculate the accuracy of the model
    logger.info("------------------------------------------------")
    logger.info("Overall Accuracy Secondary: %s",
                accuracy_score(x_test.true_secondary_class, x_test.final_secondary))
    print(precision_recall_matrix(
        x_test.true_secondary_class, x_test.final_secondary, classes))

    logger.info("Storing Secondary model output and saving...")
    # saving true tertiary classes
    save_dataset(x_test, 'secondary', id_num)

    logger.info('Completed training of secondary model %s', id_num)
    logger.debug("Elapsed time: %s", time.time() - start_time)


if __name__ == '__main__':
    main()
