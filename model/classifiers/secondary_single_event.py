"""
This module scores a single event from command line
User inputs:
body of card
subject of card
Paperless Post event_type (i.e. RSVP, LinkAway, BasicAnnouncement, ...)

Output:
Primary classification prediction and associated score
Secondary classification prediction and associated score
^^ secondary classification uses primary class as input feature ^^
"""

# pylint: disable=invalid-name

import os
import logging
import argparse
import pandas as pd

# helper modules
import context
import helpers.ml_utils
import primary_single_event
import train_secondary_model
import train_primary_model

filepath = os.path.dirname(__file__)
FEATURE_DIR = os.path.join(filepath, '../pickles/feature_pipelines/')
LOGPATH = os.path.join(filepath, '../log/event_classifier_log')

logging.basicConfig(filename=LOGPATH,
                    filemode='a',
                    format='%(asctime)s -  %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='This is the event classifier program.')

parser.add_argument('--id', help='Load existing classifier. \
        usage: --id pickles/primary_output/dataset_564.pkl', required=True)
parser.add_argument('--body', help='Body of card.', required=True)
parser.add_argument('--subject', help='Subject of card', required=True)
parser.add_argument('--event_type', help='Event_type (as stored in DB)', required=True)


def score_single_event_secondary(X, id_num):
    """
    Score a single event based on:
    Features: body, subject and event
    Trained primary classifier
    Outputs highest scoring predicted primary classification and score
    """

    primary_pred, _ = primary_single_event.score_single_event_primary(X, id_num)
    X['pred_primary_class'] = primary_pred

    clf_list, clf_names = helpers.ml_utils.load_classifier_list(id_num, 'secondary')

    feature_pipeline = helpers.ml_utils.load_classifier(FEATURE_DIR + "secondary_" + str(id_num))

    x_test_features = train_secondary_model.get_secondary_testing_features(X, feature_pipeline)

    results, classes = helpers.ml_utils.get_classifier_results(
        clf_list, clf_names, x_test_features, None)

    y_pred, y_score = train_primary_model.get_ensemble_prediction(results, classes)

    print("Secondary classification prediction and score:", y_pred, y_score)
    return y_pred, y_score


def main():
    """
    Main function that runs the classifier
    """

    logger.info("Loading existing secondary model for classification...")

    args = parser.parse_args()

    clf = args.id

    ind = clf[::-1].find('_')
    id_num = clf[::-1][4:ind][::-1]

    body = args.body
    subject = args.subject
    event_type = args.event_type

    X = pd.DataFrame([subject, body, event_type]).T
    X.columns = ['subject', 'text', 'event_type']

    score_single_event_secondary(X, id_num)
    logger.info('Completed scoring event')


if __name__ == '__main__':
    main()
