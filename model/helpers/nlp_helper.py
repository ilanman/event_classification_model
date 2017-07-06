# NLP helpers

import string
from sklearn.base import TransformerMixin
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from spacy.en import English
PARSER = English()

# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] +
               list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
REMOVE_PUNCTUATION = str.maketrans('', '', string.punctuation + "”" + "•")


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
        lemmas.append(tok.lemma_.lower().strip()
                      if tok.lemma_ != "-PRON-" else tok.lower_)
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

        # Return the classifier
        return self

    def get_params(self, deep=True):
        """
        Scikit requires a get_params function
        """
        return {}
