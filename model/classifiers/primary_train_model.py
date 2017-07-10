"""
This module trains a primary classifier
Splits a dataset into training and testing
Inputs are:
    Query of events:
        subject, body, event_type, primary_class
Ouputs:
    DataFrame with the following columns:
        subject, body, event_type, predicted_primary_class, predicted_primary_score
"""

# pylint: disable=invalid-name

import os
import time
import logging

# helper modules

import context
from helpers.ml_utils import (pickle_classifier, precision_recall_matrix, _check_col_dimension,
                              _check_row_dimension, get_classifier_results, grid_search,
                              get_ensemble_prediction, save_feature_pipeline, save_classifier,
                              save_dataset, _check_prediction_dimensions, CLASSIFIER_PIPELINE)
from helpers.nlp_helper import CleanTextTransformer, tokenize_text
from helpers.query_events import execute_query

import pandas as pd
import numpy as np

# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from scipy.sparse import hstack

filepath = os.path.dirname(__file__)
CLASSIFIER_DIR = os.path.join(filepath, '../pickles/classifiers/')
FEATURE_DIR = os.path.join(filepath, '../pickles/feature_pipelines/')
HIERARCHY_DIR = os.path.join(filepath, '../pickles/hierarchy.csv')
LOGPATH = os.path.join(filepath, '../log/event_classifier_log')

logging.basicConfig(filename=LOGPATH,
                    filemode='a',
                    format='%(asctime)s -  %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Getting event text for classifier
QUERY_ALL = """

SELECT event_id
, p_class
, s_class
, t_class
, event_name as event_name
, event_type as event_type
, event_host as event_host
, event_subject as event_subject
, text_paper as event_text
, created
FROM (
    SELECT ce.event_id
    , CASE WHEN ce.p_class = 'skip' THEN 'other' ELSE ce.p_class END
    , CASE WHEN ce.s_class = 'skip' THEN 'other' ELSE ce.s_class END
    , CASE WHEN ce.t_class = 'skip' THEN 'other' ELSE ce.t_class END
    , e.name as event_name
    , e.type as event_type
    , e.host as event_host
    , e.subject as event_subject
    , listagg(TRIM(lower(cat.text))) as text_paper
    , ce.created_at as created
    FROM event_training_selections ce
    JOIN events e ON e.id = ce.event_id
    JOIN cards c ON c.event_id = ce.event_id
    JOIN card_sides cs ON cs.card_id = c.id
        AND cs.side_type_id = 0
    LEFT JOIN card_assets cat ON cat.card_side_id = cs.id
        AND cat.asset_type_id = 9
    WHERE ce.is_confirmed
    GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 10
    )
WHERE len(trim(event_name || ' ' || event_host || ' ' || event_subject ||
        ' ' || text_paper))
ORDER BY random()

"""

PRIMARY_FEATURE_PIPELINE = dict({
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
    'event_type_pipe': Pipeline([('vectorizer', CountVectorizer())])
})


def _check_event_types(df):
    """
    Check that all event_types are one of the permissible types
    """

    acceptable_event_types = np.array([
        'BasicAnnouncement', 'DatedAnnouncement', 'GreetingCard', 'LinkAway', 'RsvpEvent'])

    for i in df.event_type.values:
        assert (i in acceptable_event_types) is True, "Found event type that doesn't exist"


def _check_primary_classes(df):
    """
    Check that all primary classes are one of the permissible types
    """

    acceptable_primary_classes = np.array([
        'birthday_celebration', 'greetings', 'organizations', 'other', 'personal', 'wedding_related'
        ])

    for i in df.p_class.values:
        assert (i in acceptable_primary_classes) is True, "Found event type that doesn't exist"


def _check_null(df):
    """
    Ensure query doesn't return any NULL values
    """

    assert np.sum(pd.isnull(df).values) == 0, "Some NULL values exist"


def _clean_df(df):
    """
    Perform any necessary cleaning

    1) Remove other class types
    2) ensure no NULL values
    3) ensure event_types are corrects
    4) ensure primary classes are correct

    """

    df[~(df.s_class == 'other')].copy()

    _check_null(df)
    _check_event_types(df)
    _check_primary_classes(df)

    return df


def get_X_and_y(df):
    """
    Input: dataframe based on query
    Output: X and y (type dataframe)
    """

    X = pd.DataFrame([df.event_subject, df.event_text, df.event_type]).T
    X.columns = ['subject', 'text', 'event_type']

    y = pd.DataFrame([df.p_class, df.s_class, df.t_class]).T
    y.columns = ['p_class', 's_class', 't_class']

    assert X.shape[0] == y.shape[0], 'X and y must be of same dimension'
    assert X.shape[1] == 3, 'X should have 3 features exactly'

    return X, y


def get_primary_training_features(X, y, feature_pipeline):
    """
    Combines training features
    Using fit_transform on the pipeline object for each feature
    Performs sparse matrix concatenation
    """

    logger.info("Beginning pipeline fit_transform to training data...")

    subject_matrix = feature_pipeline['subject_pipe'].fit_transform(X.subject, y)
    text_matrix = feature_pipeline['text_pipe'].fit_transform(X.text, y)
    event_type_matrix = feature_pipeline['event_type_pipe'].fit_transform(X.event_type, y)

    X = hstack([subject_matrix, text_matrix, event_type_matrix])

    logger.info("Completed fit_transform")

    _check_row_dimension(X, subject_matrix, text_matrix, event_type_matrix)
    _check_col_dimension(X, subject_matrix, text_matrix, event_type_matrix)

    logger.info("Training set dimension: %s", X.shape)

    return X, feature_pipeline


def get_primary_testing_features(X, feature_pipeline):
    """
    Combines training features
    Key difference between this and training_features
    is that pipeline is transforming x_test not fit_transforming
    """

    logger.info("Beginning transform of test set...")
    subject_matrix = feature_pipeline['subject_pipe'].transform(X.subject)
    text_matrix = feature_pipeline['text_pipe'].transform(X.text)
    event_type_matrix = feature_pipeline['event_type_pipe'].transform(X.event_type)

    X = hstack([subject_matrix, text_matrix, event_type_matrix])

    logger.info("Completed transform of test set.")

    _check_row_dimension(X, subject_matrix, text_matrix, event_type_matrix)
    _check_col_dimension(X, subject_matrix, text_matrix, event_type_matrix)

    logger.info("Testing set dimension: %s", X.shape)

    return X


def decorate_with_primary(X, *args):
    """
    Adds two columns to x_test
        predicted primary class
        predicted primary score
    Used as input to secondary classifier
    """

    # uses output of primary model
    X['true_primary_class'] = args[0]
    X['predicted_primary_class'] = args[1]
    X['predicted_primary_score'] = args[2]
    X['true_secondary_class'] = args[3]
    X['true_tertiary_class'] = args[4]

    return X


def save_hierarchy():
    """
    Save Hierarchy to file
    For use in enforcing hierarchy in secondary and tertiary models
    """

    df = execute_query(QUERY_ALL)
    df = df[['p_class', 's_class', 't_class']].drop_duplicates()
    df.to_csv(HIERARCHY_DIR)


def get_primary_inputs():
    """
    Query the database and return X and y variables
    """

    df = execute_query(QUERY_ALL)
    df = _clean_df(df)

    save_hierarchy()

    X, y = get_X_and_y(df)
    logger.info("Size of X input: %s", X.shape)

    return X, y,


def main():
    """
    Main function that kicks off training
    """

    start_time = time.time()
    logger.info("Training new primary model...")

    X, y = get_primary_inputs()

    x_train, x_test, y_train, y_test, _, idx2 = train_test_split(
        X, y.p_class, X.index, test_size=0.4)

    # saving for future models
    y_secondary = y.s_class[idx2]
    y_tertiary = y.t_class[idx2]

    x_train_matrix, feature_pipeline = get_primary_training_features(
        x_train, y_train, PRIMARY_FEATURE_PIPELINE)

    x_test_matrix = get_primary_testing_features(x_test, feature_pipeline)

    id_num = save_feature_pipeline(feature_pipeline, 'primary', None)

    clf_list, clf_names = grid_search(x_train_matrix, y_train, CLASSIFIER_PIPELINE)

    results, classes = get_classifier_results(clf_list, clf_names, x_test_matrix, y_test)
    save_classifier(clf_list, 'primary', id_num)

    y_pred, y_score = get_ensemble_prediction(results, classes)
    _check_prediction_dimensions(y_test, y_pred, y_score)

    # Calculate the accuracy of the model
    logger.info("------------------------------------------------")
    logger.info("Overall Accuracy Primary: %s", accuracy_score(y_test, y_pred))
    print(precision_recall_matrix(y_test, y_pred, classes))

    logger.info("Storing Primary model output and saving...")
    # saving true secondary classes
    x_test = decorate_with_primary(x_test, y_test, y_pred, y_score, y_secondary, y_tertiary)
    save_dataset(x_test, 'primary', id_num)

    logger.info('Completed training of primary model %s', id_num)
    logger.debug("Elapsed time: %s", time.time() - start_time)


if __name__ == '__main__':
    main()
