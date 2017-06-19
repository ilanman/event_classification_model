"""
This is the event classifer
Ilan Man
June 15, 2017
"""

# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name
# pylint: disable=logging-format-interpolation

import os
import sys
import string
from datetime import datetime
import time
import itertools
import logging
from ml_utils import pickle_classifier, load_classifier
from nltk.corpus import stopwords
import psycopg2
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.cm import Blues

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion, BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD

# Clean card text
from spacy.en import English
PARSER = English()

logname = 'log/event_classifier_log'
logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s -  %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

filepath = os.path.dirname(__file__)
CLASSIFIER_DIR = os.path.join(filepath, 'classifiers/')

# connect to old cluster
HOSTNAME = '10.69.10.122'
USERNAME = 'iman'
PASSWORD = os.environ['DATABASEPASSWORD']
DATABASE = 'dev'

# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
REMOVE_PUNCTUATION = str.maketrans('', '', string.punctuation + "”" + "•")
# Getting event text for classifier
QUERY = """

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
		--AND cs.num = 0
	LEFT JOIN card_assets cat ON cat.card_side_id = cs.id
		AND cat.asset_type_id = 9
	WHERE ce.is_confirmed
	GROUP BY 1, 2, 3, 4, 5, 6, 7, 9
    )
WHERE len(trim(event_name || ' ' || event_host || ' ' || event_subject || ' ' || text_paper))
ORDER BY random()

"""


def execute_query(query):
    """
    Accepts query string and connection
    Returns a dataframe with results - exactly as it appears in the DB
    """

    myConnection = psycopg2.connect(host=HOSTNAME,
                                    user=USERNAME,
                                    password=PASSWORD,
                                    dbname=DATABASE,
                                    port=5439)
    # run query and store results in a dataframe
    logger.info("Running query...")

    cur = myConnection.cursor()

    cur.execute(query)

    df_query = pd.DataFrame(cur.fetchall(), columns=['id', 'p_class', 's_class', 't_class',
                                                     'event_name', 'event_host', 'event_subject',
                                                     'event_text', 'created'])
    logger.debug("Size of dataset: {}".format(df_query.shape))
    myConnection.close()

    return df_query


def clean_text(text):
    """
    A custom function to clean the text before sending it into the vectorizer
    """

    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")

    # lowercase
    text = text.lower()

    # remove punctuation from text
    text = text.translate(REMOVE_PUNCTUATION)

    return text


def tokenize_text(sample):
    """
    A custom function to tokenize the text using spaCy and convert to lemmas

    """

    # get the tokens using spaCy
    tokens = PARSER(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # remove digits - don't provide meaning
    # but we keep things like "10th" or "11pm"
    # we lose "11 pm" and just keep "pm"
    tokens = [tok for tok in tokens if not tok.isdigit()]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")

    return tokens


class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """

    def __init__(self):
        """
        Init function
        """

        pass

    def transform(self, X):
        """
        Transform function does the work
        """
        return [clean_text(text) for text in X]

    def fit(self, X, y):
        """
        Scikit requires a fit function
        """
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)
        # Store the classes seen during fit

        # Return the classifier
        return self

    def get_params(self, deep=True):
        """
        Scikit requires a get_params function
        """
        return {}


class ExtractFeature(BaseEstimator, TransformerMixin):
    """
    Extract the correct feature to perform pipeline operations on
    """

    def __init__(self, key):
        """
        Returns the key
        """
        self.key = key

    def fit(self, X, y):
        """
        Scikit requires a fit function
        """

        #X, y = check_X_y(X, y)

        return self

    def transform(self, df):
        """
        Returns subset of dataframe where column == k
        """
        return df[self.key]


def combine_features(feat1, feat2):
    """
    Perform a Feature Union
    Takes two features - feat1 and feat2 - and applies the feature extraction, tokenization, tf-idf
    and recombines them using the FeatureUnion
    """

    combined_features = FeatureUnion([
        (feat1, Pipeline([
            ('selector', ExtractFeature(key=feat1)),
            ('cleanText', CleanTextTransformer()),
            ('vectorizer', CountVectorizer(tokenizer=tokenize_text, ngram_range=(1, 1))),
            ('tfidf', TfidfTransformer())
            ])),
        (feat2, Pipeline([
            ('selector', ExtractFeature(key=feat2)),
            ('cleanText', CleanTextTransformer()),
            ('vectorizer', CountVectorizer(tokenizer=tokenize_text, ngram_range=(1, 1))),
            ('tfidf', TfidfTransformer()),
            ('svd', TruncatedSVD(n_components=100)),
            ])),
    ])

    return combined_features


def grid_search(X, y):
    """
    Perform a Grid Search over the space of classifiers and their associated parameter space
    Inputs: X and y training sets
    Output: A list of the best classifiers from each classifier category
    """

    logger.info("Beginning Feature Union...")
    feature_union = combine_features('subject', 'text')
    # Use combined features to transform dataset
    feature_union.fit(X, y).transform(X)
    logger.info("Completed Feature Union...")

    gridsearch_pipeline = {
        'SVM': {
            'classifier': Pipeline([
                ("features", feature_union),
                ("clf", SVC(probability=True)),
            ]),
            'params' : {
                'clf__C': [1, 3],
                'clf__kernel': ['linear']
            }
        },
        'LogisticRegression' : {
            'classifier': Pipeline([
                ("features", feature_union),
                ('clf', LogisticRegression(multi_class='multinomial',
                                           class_weight='balanced',
                                           solver='lbfgs',
                                           penalty='l2'))
                ]),
            'params' : {
                'clf__C': [1, 2]
            }
        },
        'DecisionTreeClassifier' : {
            'classifier' : Pipeline([
                ("features", feature_union),
                ('clf', DecisionTreeClassifier(class_weight='balanced'))
                ]),
            'params' : {
                'clf__min_samples_split' : [2],
                'clf__min_samples_leaf' : [2],
            }
        },
    }

    logger.info("starting Gridsearch...")

    best_classifiers = []
    names = []

    for v in gridsearch_pipeline.items():
        gs = GridSearchCV(v[1]['classifier'], v[1]['params'], verbose=2)
        gs = gs.fit(X, y)
        names.append(v[0])
        logger.info("{} finished".format(v[0]))
        logger.info("Best scoring classifier: {}".format(gs.best_score_))
        best_classifiers.append(gs.best_estimator_)

    return best_classifiers, names


def most_common(lst):
    """
    Return most common element if it exists, else return False (indicating a tie)
    """

    if len(np.unique(lst)) == len(lst):
        return 'no_majority'

    return max(set(list(lst)), key=list(lst).count)


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

    num_clfs = len(results)

    # combine all classifier predictions and probabilities
    all_preds = np.array([v[0] for k, v in results.items()]).T
    all_probs = np.sum(np.array([v[1] for k, v in results.items()]), axis=0)

    # find elements where majority exists - the function most_common returns majority class or
    # "no_majority"
    majority = np.array(list(map(most_common, all_preds)))
    no_majority_index = np.where(majority == 'no_majority')

    # for those where a majority doesn't exist, sum the probabilities for each class
    no_majority_sum = all_probs[no_majority_index]

    # ensure no new probabilities were added that shouldn't be
    assert np.sum(no_majority_sum)/num_clfs == len(no_majority_index[0]), \
            "probability sum is greater than expected for no_majority"

    # replace the "no_majority" samples with the class that resulted in the largest probability
    majority[no_majority_index] = classes[np.argmax(no_majority_sum, axis=1)]

    return majority


def precision_recall_matrix(result, labels):
    """
    Use results of classifiers to create a precision and recall matrix
    """

    ## Create the precision and recall matrix
    labels = ['greetings', 'other', 'organizations', 'birthday_celebration', 'wedding_related',
              'personal']

    result_prec_recall = precision_recall_fscore_support(y_test, y_pred, labels=labels)
    result = pd.DataFrame()
    result['classification'] = labels
    result['precision'] = result_prec_recall[0]
    result['recall'] = result_prec_recall[1]

    # Display the precision and recall table
    logger.info(result)


#def plot_confusion_matrix(cm, classes, title):
#    """
#    This function prints and plots the confusion matrix
#    """
#    plt.imshow(cm, interpolation='nearest', cmap=Blues)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, cm[i, j],
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#

def get_classifier_results(gs, clf_names, x_test, y_test, save):
    """
    Stores results of each classifier in a dictionary
    First element is the predicted class, clf.predict()
    Second element is the probability, clf.predict_proba()
    Ensure that each classifier passed has a predict_proba() method

    Prints the accuracy acheived for each classifier
    Returns a dictionary of results and the class labels
    """

    results = {}

    if not to_save:
        gs = [gs]

    classes = gs[0].classes_
    clf_time = datetime.today()

    for (name, clf) in zip(clf_names, gs):
        results[name] = [clf.predict(x_test), clf.predict_proba(x_test)]
        logger.info("------------------------------------------------")
        logger.info("{} Accuracy: {}".format(name, accuracy_score(y_test, results[name][0])))
        if save:
            pickle_classifier(clf, CLASSIFIER_DIR + name + "_" + clf_time.strftime('%m%d%Y%H%M%S'))

    return results, classes


def get_input_variables(df):
    """
    Input: dataframe of results
    Output: training and testing variables
    """

    X = pd.DataFrame([df.event_subject, df.event_text]).T
    X.columns = ['subject', 'text']
    y = df.p_class.values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':

    start_time = time.time()
    logger.info("Running classifier...")

    df = execute_query(QUERY)

    x_train, x_test, y_train, y_test = get_input_variables(df)


    if len(sys.argv) == 1:
        to_save = True
        print("Building new classifiers...")
        gs, clf_names = grid_search(x_train, y_train)
    else:
        to_save = False
        clf_id = sys.argv[1]
        print("Running {} on test set".format(clf_id))
        gs, clf_names = load_classifier(CLASSIFIER_DIR + clf_id), [clf_id[1:5]]

    # get_classifier iterates over a list, so ensure params are correct type
    results, classes = get_classifier_results(gs, clf_names, x_test, y_test, to_save)

    y_pred = get_ensemble_prediction(results, classes)

    def sliceit(x):
        """
        Need to investigate why sci kit removes last few characters of class
        This only looks at first 5 elements for accuracy/comparison
        """
        return x[:5]

    y_pred = list(map(sliceit, y_pred))
    y_test = list(map(sliceit, y_test))

    # Calculate the accuracy of the model
    logger.info("------------------------------------------------")
    logger.info("Overall Accuracy:{}".format(accuracy_score(y_test, y_pred)))

    logger.debug("Elapsed time:{}".format(time.time() - start_time))

    # Compute confusion matrix
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    #np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=classes, title='Confusion matrix')
