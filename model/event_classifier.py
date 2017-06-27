"""
This is the event classifer
Ilan Man
June 23, 2017
"""

import os
import time
from collections import Counter
import logging
import argparse

# helper modules
from ml_utils import pickle_classifier, load_classifier, ExtractFeature
from nlp_helper import CleanTextTransformer, tokenize_text
from query_events import execute_query

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

logname = 'log/event_classifier_log'
logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s -  %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='This is the event classifier program.')
parser.add_argument('--level', help='Level of classification, \
        usage: --level primary', choices=["primary", "secondary", "tertiary"], required=True)
parser.add_argument('--retrain', help='Retrain classifier. usage: --retrain F',
                    choices=['T', 'F'], required=True)
parser.add_argument('--load_clf', help='Load existing classifier. \
        usage: --load_clf classifiers/SVM_06202017121413.pkl', nargs='+', required=False)
parser.add_argument('--event_ids', help='Enter events to classify. If blank \
        then query will fetch all training data. usage: --event_ids 998746 \
        33384956 114992', nargs='+', required=False)
filepath = os.path.dirname(__file__)
CLASSIFIER_DIR = os.path.join(filepath, 'classifiers/')

MAPPING = {
    'birthday_celebration': ['kids_birthday', 'adult_birthday', 'general'],
    'personal': ['announcements_personal', 'adult_or_family_events',
                 'celebrating_baby_kids_or_parents_to_be',
                 'other_holiday_events', 'seasonal_holiday_events'],
    'wedding_related': ['wedding_events_hosted_by_bride_groom_or_family',
                        'wedding_related_other', 'save_the_date',
                        'parties_and_showers_in_honor_of_the_bride_&_groom'],
    'organizations': ['alumni_or_school_related', 'business_or_nonprofits'],
    'other': ['foreign_language', 'address_collection'],
    'greetings': ['everyday_greetings', 'seasonal_holiday_cards',
                  'other_holiday_cards']
    }

# Getting event text for classifier
QUERY_ALL = """

SELECT event_id
, p_class
, s_class
, t_class
, event_name as event_name
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
    GROUP BY 1, 2, 3, 4, 5, 6, 7, 9
    )
WHERE len(trim(event_name || ' ' || event_host || ' ' || event_subject ||
        ' ' || text_paper))
ORDER BY random()

"""

QUERY_EVENTS = """

SELECT event_id
, event_name as event_name
, event_host as event_host
, event_subject as event_subject
, text_paper as event_text
, created
FROM (
    SELECT e.id as event_id
    , e.name as event_name
    , e.host as event_host
    , e.subject as event_subject
    , listagg(TRIM(lower(cat.text))) as text_paper
    , e.created_at as created
    FROM events e
    JOIN cards c ON c.event_id = e.id
    JOIN card_sides cs ON cs.card_id = c.id
        AND cs.side_type_id = 0
    LEFT JOIN card_assets cat ON cat.card_side_id = cs.id
        AND cat.asset_type_id = 9
    WHERE e.id LIKE {0}
    GROUP BY 1, 2, 3, 4, 6
    )
WHERE len(trim(event_name || ' ' || event_host || ' ' || event_subject ||
        ' ' || text_paper))
ORDER BY random()

"""


def combine_secondary_features(feat1, feat2, feat3):
    """
    Perform a Feature Union
    Takes two features - feat1 and feat2 - and applies the feature extraction,
    tokenization, tf-idf
    and recombines them using the FeatureUnion
    """

    combined_features = FeatureUnion([
        (feat1, Pipeline([
            ('selector', ExtractFeature(key=feat1)),
            ('cleanText', CleanTextTransformer()),
            ('vectorizer', CountVectorizer(tokenizer=tokenize_text,
                                           ngram_range=(1, 1))),
            ('tfidf', TfidfTransformer())
            ])),
        (feat2, Pipeline([
            ('selector', ExtractFeature(key=feat2)),
            ('cleanText', CleanTextTransformer()),
            ('vectorizer', CountVectorizer(tokenizer=tokenize_text,
                                           ngram_range=(1, 1))),
            ('tfidf', TfidfTransformer()),
            # ('svd', TruncatedSVD(n_components=100)),
            ])),
        (feat3, Pipeline([
            ('selector', ExtractFeature(key=feat3)),
            ('onehot', CountVectorizer()),
            ])),
    ])

    return combined_features


def combine_primary_features(feat1, feat2):
    """
    Perform a Feature Union
    Takes two features - feat1 and feat2 - and applies the feature extraction,
    tokenization, tf-idf
    and recombines them using the FeatureUnion
    """

    combined_features = FeatureUnion([
        (feat1, Pipeline([
            ('selector', ExtractFeature(key=feat1)),
            ('cleanText', CleanTextTransformer()),
            ('vectorizer', CountVectorizer(tokenizer=tokenize_text,
                                           ngram_range=(1, 1))),
            ('tfidf', TfidfTransformer())
            ])),
        (feat2, Pipeline([
            ('selector', ExtractFeature(key=feat2)),
            ('cleanText', CleanTextTransformer()),
            ('vectorizer', CountVectorizer(tokenizer=tokenize_text,
                                           ngram_range=(1, 1))),
            ('tfidf', TfidfTransformer()),
            ])),
    ])

    return combined_features


def get_features(x_train, y_train, level, x_test, train_model, id_num):
    """
    Transform raw data into inputs for the model
    If the data is training, then fit the feature union
    If the input data is testing, then only transform
    Return feature union
    """

    logger.info("Beginning {} Feature Union....".format(level))

    if level == 'primary':
        feature_union = combine_primary_features(
            'subject', 'text')
    else:
        feature_union = combine_secondary_features(
            'subject', 'text', 'pred_p_class')

    # need to retrain a model
    if train_model:
        x_train = feature_union.fit_transform(x_train, y_train)

        if id_num is None:
            id_num = np.random.randint(1000)

        pickle_classifier(
            feature_union, CLASSIFIER_DIR + "_" + level + "_features_" + str(id_num))

    # load existing feature union
    else:
        feature_union = load_classifier(
            CLASSIFIER_DIR + "_" + level + "_features_" + str(id_num))

    x_test = feature_union.transform(x_test)

    logger.info("Completed Feature Union...")

    return x_train, x_test, id_num


def grid_search(X, y):
    """
    Perform a Grid Search over the space of classifiers and their associated
    parameter space
    Inputs: X and y training sets
    Output: A list of the best classifiers from each classifier category
    """

    gridsearch_pipeline = {
        'SVM': {
            'classifier': Pipeline([
                ("clf", SVC(probability=True)),
            ]),
            'params': {
                'clf__C': [3],
                'clf__kernel': ['linear']
            }
        },
        'LogisticRegression': {
            'classifier': Pipeline([
                ('clf', LogisticRegression(multi_class='multinomial',
                                           class_weight='balanced',
                                           solver='lbfgs',
                                           penalty='l2'))
                ]),
            'params': {
                'clf__C': [1]
            }
        },
    }

    logger.info("starting Gridsearch...")

    best_classifiers = []
    names = []

    for v in gridsearch_pipeline.items():
        gs = GridSearchCV(v[1]['classifier'], v[1]['params'], verbose=0, cv=3,
                          n_jobs=4)
        gs = gs.fit(X, y)
        names.append(v[0])
        logger.info("{} finished".format(v[0]))
        logger.info("Best scoring classifier: {}".format(gs.best_score_))
        best_classifiers.append(gs.best_estimator_)

    return best_classifiers, names


def most_common(lst):
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

    ex.
    clf_1 = [0.1,0.1,0.8]  - choose 3
    clf_2 = [0.1,0.8,0.1]  - choose 2
    clf_3 = [0.5,0.3,0.2]  - choose 1
    no majority exists

    add probabilities
    tiebreaker = [0.7,1.2,1.1] - choose 2
    """

    print("Getting ensemble predictions...")
    num_clfs = len(results)

    # combine all classifier predictions and probabilities
    all_preds = np.array([v[0] for k, v in results.items()]).T
    all_probs = np.sum(np.array([v[1] for k, v in results.items()]), axis=0)
    all_probs_normalize = all_probs/num_clfs

    # the function most_common returns majority class or "no_majority"
    majority = np.array(list(map(most_common, all_preds)))
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


def precision_recall_matrix(y_test, y_pred, classes):
    """
    Use results of classifiers to create a precision and recall matrix
    """

    #  Create the precision and recall matrix
    result_prec_recall = precision_recall_fscore_support(y_test, y_pred,
                                                         labels=classes)
    result = pd.DataFrame()
    result['classification'] = classes
    result['precision'] = result_prec_recall[0]
    result['recall'] = result_prec_recall[1]

    # Display the precision and recall table
    return result


def get_classifier_results(gs, clf_names, x_test, y_test):
    """
    Stores results of each classifier in a dictionary
    First element is the predicted class, clf.predict()
    Second element is the probability, clf.predict_proba()
    Ensure that each classifier passed has a predict_proba() method

    Prints the accuracy acheived for each classifier
    Returns a dictionary of results and the class labels
    """

    print("Getting classifier results...")
    results = {}
    classes = gs[0].classes_

    for (name, clf) in zip(clf_names, gs):

        results[name] = [clf.predict(x_test), clf.predict_proba(x_test)]
        logger.info("------------------------------------------------")
        if y_test is not None:  # if there is a testing set
            logger.info("{} Accuracy: {}".format(
                name, accuracy_score(y_test, results[name][0])))

    return results, classes


def get_Xy(df):
    """
    Returns X (type: dataframe) and y (type: dictionary)
    """

    df = df[~(df.s_class == 'other')].copy()  # remove "other"
    X = pd.DataFrame([df.event_subject, df.event_text]).T
    X.columns = ['subject', 'text']

    y = {
        'primary': df.p_class,
        'secondary': df.s_class,
        'tertiary': df.t_class
        }

    assert X.shape[0] == len(y['primary']), 'X and y must be of same dimension'
    return X, y


def get_input_variables(X, y, level, test_size):
    """
    Returns the train and test split
    """

    x_train, x_test, y_train, y_test, idx1, idx2 = \
        train_test_split(X, y[level], X.index, test_size)

    return x_train, x_test, y_train, y_test, idx1, idx2


def enforce_hierarchy(row, single_event=False):
    """
    # use primary classification as feature input
    # get secondary classification
     # when secondary is part of primary hierarchy, exit
     # when secondary is not part, see which score was higher
      # if primary score higher, keep primary class, set secondary to unknown
      # if secondary score higher, keep secondary class, set primary
    # to what it should be (enforce hierarchy)
    """

    try:

        if single_event:

            primary, secondary, p_score, s_score = \
                row['pred_p_class'].values[0], \
                row['pred_s_class'].values[0], \
                row['pred_p_score'].values[0], \
                row['pred_s_score'].values[0]

        else:

            primary, secondary, p_score, s_score = \
                row['pred_p_class'], \
                row['pred_s_class'], \
                row['pred_p_score'], \
                row['pred_s_score']

        if secondary in MAPPING[primary]:  # hierarchy is enforced
            return primary, secondary

        # model is more confident in primary than secondary
        if p_score > s_score:
            return primary, 'unknown'   # remove secondary, keep primary

        for k, v in MAPPING.items():
            if secondary in v:
                return k, secondary   # keep secondary, remove primary

        return

    except ValueError as e:
        print(e)
        print(row)


def check_hierarchy(X):
    """
    Check that the hierarchy is enforced as expected
    """

    unknown_list = []
    for row in X.iterrows():
        if row[1].final_secondary not in MAPPING[row[1].final_primary]:
            unknown_list.append(row[1].final_secondary)

    if unknown_list != []:
        assert sum(np.array(unknown_list) != 'unknown') == 0, (
               "Hierarchy not enforced")


def add_final_classes(X):
    """
    Enforce the hierarchy and add final classification
    """

    print("Adding final classifications to dataframe...")
    print("Shape of final X:", X.shape)
    # convert to 2D array and apply hierarchy
    # if X is one record, the syntax changes slightly
    if X.shape[0] > 1:

        finals = X.apply(enforce_hierarchy, axis=1)
        finals = np.array([np.array([i[0], i[1]]) for i in finals])

        X.loc[:, 'final_primary'] = finals[:, 0]
        X.loc[:, 'final_secondary'] = finals[:, 1]

    else:
        finals = enforce_hierarchy(X, single_event=True)

        X['final_primary'] = finals[0]
        X['final_secondary'] = finals[1]

    return X


def get_secondary_classifier_training_set(X, y_secondary, y_pred_primary,
                                          y_score_primary, idx2):
    """
    Inputs: X training set
            true secondary classification vector - this is the target
            predicted primary classification - additional feature
            predicted primary scores
            indexes to use for training (which correspond to the indexes that
            the primary model tested)
    """

    logger.info("Running secondary classifier on primary classifier results..")

    # set up secondary model
    # uses output of primary model
    X['pred_p_class'] = y_pred_primary
    X['pred_p_score'] = y_score_primary

    print("training X shape: ", X.shape)

    x_train, x_test, y_train, y_test, _, _ = train_test_split(
        X, y_secondary, idx2, test_size=0.5)

    return x_train, x_test, y_train, y_test


def decorate_with_primary(X, y_pred_primary, y_score_primary):
    """
    Inputs: X training set
            predicted primary classification - additional feature
            predicted primary scores
    """

    logger.info("Running secondary classifier on primary classifier results..")

    # uses output of primary model
    X['pred_p_class'] = y_pred_primary
    X['pred_p_score'] = y_score_primary

    return X


def get_final_predictions(X, y_pred, y_score, y_primary):
    """
    Append final results to the x_test data frame
    Enforce a hierarchical structure
    """

    print("Getting final predictions...")
    X['pred_s_class'] = y_pred
    X['pred_s_score'] = y_score

    if y_primary is not None:
        X['true_p_class'] = y_primary

    X = add_final_classes(X)
    check_hierarchy(X)

    return X


def load_classifier_list(clf):
    """
    Load a classifier
    Classifier is stored as list object
    Returns list of classifiers and their names
    """

    print("Loading classifier list...")
    # clean up input: remove "classifier/" and ".pkl"
    clf_id = clf[clf.find('/')+1:-4]

    clf_list, clf_names = [], []
    # this is a list of classifiers
    loaded_clf = load_classifier(CLASSIFIER_DIR + clf_id)
    for classifier in loaded_clf:
        clf_list.append(classifier)

        # get name via class structure
        clf_class = str(classifier.named_steps['clf'].__class__)

        # some basic cleaning of class name
        clf_name_indx = clf_class.find('.')
        clf_name = clf_class[clf_name_indx+1:-2]
        clf_names.append(clf_name)

    return clf_list, clf_names


def retrain_primary(X, y, level):

    logger.info("Building new primary classifier...")

    x_train, x_test, y_train, y_test, idx1, idx2 = train_test_split(
            X, y, X.index, test_size=0.4)

    x_train_feature, x_test_feature, id_num = get_features(
        x_train, y_train, level, x_test, train_model=True, id_num=None)

    gs, clf_names = grid_search(x_train_feature, y_train)

    pickle_classifier(
        gs, CLASSIFIER_DIR + "_" + level + "_classifier_" + str(id_num))

    results, classes = get_classifier_results(
        gs, clf_names, x_test_feature, y_test)

    y_pred, y_score = get_ensemble_prediction(results, classes)
    assert y_pred.shape[0] == len(idx2), (
        "Ensure prediction vector is same length as test set")

    # Calculate the accuracy of the model
    logger.info("------------------------------------------------")
    logger.info("Overall Accuracy Primary:{}"
                .format(accuracy_score(y_test, y_pred)))
    print(precision_recall_matrix(y_test, y_pred, classes))


def load_primary(X, y, level, clfs):

    primary_clf = clfs[0]

    # some string cleaning to get the id_num
    ind = primary_clf[::-1].find('_')
    id_num = primary_clf[::-1][4:ind][::-1]

    logger.info("Loading primary classifier {}...".format(primary_clf))

    x_train, x_test, y_train, y_test, idx1, idx2 = train_test_split(
            X, y, X.index, test_size=0.4)

    x_train_feature, x_test_feature, id_num = get_features(
        x_train, y_train, level, x_test, train_model=False, id_num=id_num)

    gs, clf_names = load_classifier_list(primary_clf)

    results, classes = get_classifier_results(
        gs, clf_names, x_test_feature, y_test)

    y_pred, y_score = get_ensemble_prediction(results, classes)
    assert y_pred.shape[0] == len(idx2), (
        "Ensure prediction vector is same length as test set")

    # Calculate the accuracy of the model
    logger.info("------------------------------------------------")
    logger.info("Overall Accuracy Primary:{}".
                format(accuracy_score(y_test, y_pred)))
    print(precision_recall_matrix(y_test, y_pred, classes))

    return y_pred, y_score, primary_clf, id_num, idx2, x_test


def train_secondary(X, y, level, clfs):

    y_pred, y_score, primary_clf, id_num, idx2, x_test = load_primary(
        X, y['primary'], 'primary', clfs)

    # # do 3) above
    # primary_clf = clfs[0]
    # ind = primary_clf[::-1].find('_')
    # id_num = primary_clf[::-1][4:ind][::-1]

    logger.info("Loading primary classifier {} \
        to use in training a secondary classifier".format(primary_clf))

    # index y_secondary with idx2 - the indexes used for testing primary
    y_primary = y['primary'][idx2]
    y_secondary = y['secondary'][idx2]

    # ensure dimensions of test set are consistent
    assert x_test.shape[0] == len(y_secondary)
    assert x_test.shape[0] == len(y_pred)
    assert x_test.shape[0] == len(y_score)
    assert x_test.shape[0] == len(y_primary)

    # get new training/testing set for Secondary Model
    x_train, x_test, y_train, y_test = \
        get_secondary_classifier_training_set(
            x_test, y_secondary, y_pred, y_score, idx2)

    assert x_train.shape[1] == 4, 'Check dimensions of Secondary Training'

    # create list of classifiers
    x_train_feature, x_test_feature, id_num = get_features(
        x_train, y_train, 'secondary', x_test, train_model=True, id_num=id_num)

    gs, clf_names = grid_search(x_train_feature, y_train)

    # pickle list of classifiers
    pickle_classifier(
        gs, CLASSIFIER_DIR + "_" + level + "_classifier_" + str(id_num))

    # get individual classifier predictions
    results, classes = get_classifier_results(
        gs, clf_names, x_test_feature, y_test)

    # get ensemble predictions based on ensemble rules
    y_pred, y_score = get_ensemble_prediction(results, classes)

    x_test = get_final_predictions(x_test, y_pred, y_score, y_primary)

    # Calculate the accuracy of the model
    logger.info("------------------------------------------------")
    logger.info("Overall Accuracy Primary: {}".
                format(accuracy_score(x_test['true_p_class'],
                                      x_test['final_primary'])))

    logger.info("------------------------------------------------")
    logger.info("Overall Accuracy Secondary: {}".
                format(accuracy_score(y_test, x_test['final_secondary'])))

    print(precision_recall_matrix(
        y_test, x_test['final_secondary'], classes))

    return x_test


def load_secondary(X, y, level, clfs):

    y_pred, y_score, primary_clf, id_num, idx2, x_test = load_primary(
        X, y['primary'], 'primary', clfs)

    secondary_clf = clfs[1]
    logger.info("Loading secondary classifier {}...".format(secondary_clf))

    gs, clf_names = load_classifier_list(secondary_clf)

    # index y_secondary with idx2 - the indexes used for testing primary
    y_primary = y['primary'][idx2]
    y_secondary = y['secondary'][idx2]

    # ensure dimensions of test set are consistent
    assert x_test.shape[0] == len(y_secondary)
    assert x_test.shape[0] == len(y_pred)
    assert x_test.shape[0] == len(y_score)
    assert x_test.shape[0] == len(y_primary)

    # get new training/testing set for Secondary Model
    x_train, x_test, y_train, y_test = \
        get_secondary_classifier_training_set(
            x_test, y_secondary, y_pred, y_score, idx2)

    assert x_train.shape[1] == 4, 'Check dimensions of Secondary Training'

    # create list of feature objects
    x_train_feature, x_test_feature, id_num = get_features(
        x_train, y_train, 'secondary', x_test, train_model=False, id_num=id_num)

    # get individual classifier predictions
    results, classes = get_classifier_results(
        gs, clf_names, x_test_feature, y_test)

    y_pred, y_score = get_ensemble_prediction(results, classes)

    assert x_train.shape[1] == 4, 'Check dimensions of Secondary Training'

    x_test = get_final_predictions(x_test, y_pred, y_score, y_primary)

    # Calculate the accuracy of the model
    logger.info("------------------------------------------------")
    logger.info("Overall Accuracy Primary: {}".
                format(accuracy_score(x_test['true_p_class'],
                                      x_test['final_primary'])))

    logger.info("------------------------------------------------")
    logger.info("Overall Accuracy Secondary: {}".
                format(accuracy_score(y_test, x_test['final_secondary'])))

    print(precision_recall_matrix(
        y_test, x_test['final_secondary'], classes))

    return x_test


def score_single_event(X, event_ids):

    # do 5) above

    primary_clf = clfs[0]
    secondary_clf = clfs[1]

    ind = primary_clf[::-1].find('_')
    id_num = primary_clf[::-1][4:ind][::-1]

    logger.info("Loading primary classifier {}...".format(primary_clf))
    gs, clf_names = load_classifier_list(primary_clf)

    _, x_test_feature, _ = get_features(
        None, None, 'primary', X, train_model=False, id_num=id_num)

    results, classes = get_classifier_results(
        gs, clf_names, x_test_feature, y_test=None)

    y_pred, y_score = get_ensemble_prediction(results, classes)

    logger.info("Loading secondary classifier {}...".format(secondary_clf))
    gs, clf_names = load_classifier_list(secondary_clf)

    X_second = decorate_with_primary(X, y_pred, y_score)

    _, x_test_feature, _ = get_features(
        None, None, 'secondary', X_second, train_model=False, id_num=id_num)

    results, classes = get_classifier_results(
        gs, clf_names, x_test_feature, y_test=None)

    y_pred, y_score = get_ensemble_prediction(results, classes)

    x_test = get_final_predictions(X_second, y_pred, y_score, y_primary=None)

    return x_test


if __name__ == '__main__':

    args = parser.parse_args()

    start_time = time.time()
    logger.info("Running classifier script...")

    event_ids = args.event_ids
    level = args.level
    retrain = args.retrain
    clfs = args.load_clf
    event_ids = args.event_ids

    # QUERY DATABASE AND GET X, y variables
    if event_ids is None:
        df = execute_query(QUERY_ALL, event_id=False)

        X, y = get_Xy(df)
        print("Size of dataset:", X.shape)
    else:

        df = execute_query(
            QUERY_EVENTS.format(str(event_ids[0])),
            event_id=True)
        X = pd.DataFrame([df.event_subject, df.event_text]).T
        X.columns = ['subject', 'text']
        print("---------------")
        print("Subject:")
        print(X.subject.values[0])
        print("Card text:")
        print(X.text.values[0])
        print("---------------")

    if level == 'primary' and retrain == 'T' and clfs is None \
       and event_ids is None:

        retrain_primary(X, y['primary'], level)

    elif level == 'primary' and retrain == 'F' and len(clfs) == 1:

        y_pred, y_score, primary_clf, id_num = load_primary(X, y['primary'], level, clfs)

    elif level == 'secondary' and retrain == 'T'and len(clfs) == 1 and event_ids is None:

        results = train_secondary(X, y, level, clfs)

    elif level == 'secondary' and retrain == 'F' and len(clfs) == 2 and event_ids is None:

        results = load_secondary(X, y, level, clfs)

    elif level == 'secondary' and retrain == 'F' and len(clfs) == 2 and event_ids is not None:

        results = score_single_event(X, event_ids)
        print(results)

    logger.debug("Elapsed time:{}".format(time.time() - start_time))
