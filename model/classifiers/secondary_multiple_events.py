"""
This module scores a single event from command line
User inputs:
body of card
subject of card
Paperless Post event_type (i.e. RSVP, LinkAway, BasicAnnouncement, ...)

Output:
Primary classification prediction and associated score
"""

# pylint: disable=invalid-name

import os
import logging
import argparse
import time
from datetime import datetime
import pandas as pd
import numpy as np

# helper modules
import context
from helpers import ml_utils, query_events
import primary_multiple_events
import secondary_train_model

filepath = os.path.dirname(__file__)
FEATURE_DIR = os.path.join(filepath, '../pickles/feature_pipelines/')
LOGPATH = os.path.join(filepath, '../log/event_classifier_log')
OUTPUT_CSV = os.path.join(filepath, '../csv_files/')

logging.basicConfig(filename=LOGPATH,
                    filemode='a',
                    format='%(asctime)s -  %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='This is the event classifier program.')

parser.add_argument('--id', help='Load existing classifier. \
        usage: --id pickles/classifier/secondary_564.pkl', required=True)
parser.add_argument('--start', help='Classify events starting after this date. \
        usage: --start 2017-06-01', required=True)
parser.add_argument('--end', help='Classify events starting before this date (inclusive). \
        usage: --end 2017-06-10', required=True)


QUERY_EVENTS = """

SELECT event_id
 , event_subject as event_subject
 , text_paper as event_text
 , event_type as event_type
 , created
 FROM (
     SELECT e.id as event_id
     , e.type as event_type
     , e.subject as event_subject
     , listagg(TRIM(lower(cat.text))) as text_paper
     , e.created_at as created
     FROM events e
     JOIN cards c ON c.event_id = e.id
     JOIN card_sides cs ON cs.card_id = c.id
         AND cs.side_type_id = 0
     LEFT JOIN card_assets cat ON cat.card_side_id = cs.id
         AND cat.asset_type_id = 9
     WHERE DATE(e.created_at) = '{}'
     GROUP BY 1, 2, 3, 5
     )
 WHERE len(trim(event_subject || ' ' || text_paper))
 ORDER BY random()

"""


def _check_event_types(df):
    """
    Check that all event_types are one of the permissible types
    """

    acceptable_event_types = np.array([
        'BasicAnnouncement', 'DatedAnnouncement', 'GreetingCard', 'LinkAway', 'RsvpEvent'])

    for i in df.event_type.values:
        assert (i in acceptable_event_types) is True, "Found event type that doesn't exist"


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

    _check_null(df)
    _check_event_types(df)

    return df


def score_chunks_secondary(X, id_num):
    """
    Score a single event based on:
    Features: body, subject and event
    Trained secondary classifier
    Outputs highest scoring predicted primary classification and score
    """

    primary_pred, primary_score = primary_multiple_events.score_chunks_primary(X, id_num)
    X['predicted_primary_class'] = primary_pred
    X['predicted_primary_score'] = primary_score

    # load classifier (stored as list of classifiers)
    clf_list, clf_names = ml_utils.load_classifier_list(id_num, 'secondary')
    # load feature pipeline
    feature_pipeline = ml_utils.load_classifier(FEATURE_DIR + "secondary_" + str(id_num))
    # transform data using feature pipeline
    x_test_features = secondary_train_model.get_secondary_testing_features(X, feature_pipeline)

    results, classes = ml_utils.get_classifier_results(clf_list, clf_names, x_test_features, None)

    second_pred, second_score = secondary_train_model.get_ensemble_prediction(results, classes)
    X['predicted_secondary_class'] = second_pred
    X['predicted_secondary_score'] = second_score

    X = secondary_train_model.add_final_secondary_classes(X)

    return X


def convert_df_to_sql(results, df):
    """
    Get results and append them to event_id dataframe
    """

    df_for_upload = pd.DataFrame(df.event_id, columns=['event_id'])
    df_for_upload['primary_class'] = results['final_primary']
    df_for_upload['primary_score'] = results['predicted_primary_score']
    df_for_upload['secondary_class'] = results['final_secondary']
    df_for_upload['secondary_score'] = results['predicted_secondary_score']
    df_for_upload['timestamp'] = datetime.today()

    return df_for_upload


def main():
    """
    Main function that runs the program
    """

    start_time = time.time()

    args = parser.parse_args()

    id_num = args.id

    start_date = args.start
    end_date = args.end

    table_name = 'secondary_classifications'

    logger.info("Classifying events from %s to %s...", start_date, end_date)

    dates = ml_utils.create_date_range(start_date, end_date)

    for _date in dates:
        logger.info("Classifying events created at %s...", _date)

        df = query_events.execute_query(QUERY_EVENTS.format(_date))
        df = _clean_df(df)

        X = pd.DataFrame([df.event_subject, df.event_text, df.event_type]).T
        X.columns = ['subject', 'text', 'event_type']

        results = score_chunks_secondary(X, id_num)
        for_database = convert_df_to_sql(results, df)

        query_events.create_staging_and_insert_secondary(table_name, for_database.values.tolist())
        query_events.merge_records(table_name)
        logger.info("Completed upserting %s", _date)

        # classified_events.to_csv(OUTPUT_CSV + "primary_" + _date + "_.csv", mode='w')

    logger.info('Completed secondary classifications of events from %s to %s', start_date, end_date)
    logger.info("Elapsed Time: %s", time.time() - start_time)


if __name__ == '__main__':

    main()
