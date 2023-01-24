import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd


def prep_raw_data(data):
    """
    Handler for data cleaning and preprocessing
    :param data: pandas DataFrame to prepare
    :return: cleaned and preprocessed DataFrame
    """
    data = clean_data(data)
    data = stem_lem_and_token(data)
    data = join_data_text(data)
    return data


def clean_data(data):
    """
    Cleans dataset features (text) by removing stopwords, punctuation, repeated characters, and other irrelevant
    text articles
    :param data: pandas DataFrame to clean
    :return: DataFrame with cleaned feature texts
    """
    new_data = data.copy()
    # lower case only
    new_data['x'] = new_data['x'].str.lower()
    # remove stop words
    download_nltk_stopwords()
    download_nltk_wordnet()
    stop_words = stopwords.words('english')
    stop_words.append('BACKGROUND')  # custom word removal
    new_data['x'] = new_data['x'].apply(lambda text: remove_stopwords(text, stop_words))
    # separate punctuation, could replace instead
    rem_punc = string.punctuation
    keep_punc = '!?'
    new_data['x'] = new_data['x'].apply(lambda text: separate_punctuation(text, rem_punc, keep_punc))
    # remove repeating characters
    new_data['x'] = new_data['x'].apply(lambda text: remove_repeated_character(text))
    # remove misc words: email addresses, URL, numerics
    new_data['x'] = new_data['x'].apply(lambda text: remove_misc(text))
    return new_data


def download_nltk_stopwords():
    """
    Downloads NLTK stopwords if not already present
    """
    try:
        nltk.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('omw-1.4')


def download_nltk_wordnet():
    """
    Downlaods NLTK wordnet if not already present
    """
    try:
        lemmatizer = nltk.WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(word) for word in 'hello']
    except LookupError:
        nltk.download('wordnet')


def stem_lem_and_token(data):
    """
    Applies tokenisation, stemming and lemmatisation to data features
    :param data: pandas DataFrame to prepare
    :return: prepared DataFrame
    """
    new_data = data
    # tokenise
    tokenizer = RegexpTokenizer(r'\w+')
    new_data['x'] = new_data['x'].apply(tokenizer.tokenize)
    # stemming
    new_data['x'] = new_data['x'].apply(lambda text: stemming(text))
    # lemmatization
    new_data['x'] = new_data['x'].apply(lambda text: lemmatizer(text))
    return new_data


def join_data_text(data_split):
    """
    Joins text back together after stemming and lemmatisation
    :param data_split: list of words
    :return: joined text string
    """
    new_data = data_split.copy()
    new_data['x'] = new_data['x'].apply(lambda text: ' '.join(text))
    return new_data


def remove_stopwords(text, stop_words):
    """
    Removes a set of stopwords from a text
    :param text: raw text string
    :param stop_words: list of stopwords to remove
    :return: new text string without stopwords
    """
    return ' '.join([word for word in str(text).split() if word not in stop_words])


def separate_punctuation(text, rem_punc, keep_punc):
    """
    Removes punctuation from text
    :param text: raw text
    :param rem_punc: punctuation to remove
    :param keep_punc: punctuation to retain
    :return: new text without undesired punctuation
    """
    punctuation = rem_punc.translate(str.maketrans('', '', keep_punc))
    translator = str.maketrans('', '', punctuation)
    # remove unwanted punctuation
    new_text = text.translate(translator)
    # split wanted punctuation
    new_text = re.sub(r'[^\w\s]', '', new_text)
    return new_text


def remove_repeated_character(text):
    """
    Removes repeated characters, particularly for non-standard English
    :param text: raw text string
    :return: cleaned text
    """
    return re.sub(r'(.)\1+', r'\1', text)


def remove_misc(text):
    """
    Removes tags, URLs, and numbers from a text
    :param text: raw text string
    :return: new text without undesired articles
    """
    new_text = re.sub('@[^\s]+', '', text)
    new_text = re.sub('((www.[^s]+)|(https?://[^s]+))', '', new_text)
    new_text = re.sub('[0-9]+', '', new_text)
    return new_text


def stemming(text):
    """
    Applies stemming to text to reduce variations of core words
    :param text: raw text string
    :return: stemmed text string
    """
    stemmer = nltk.PorterStemmer()
    stemmed = [stemmer.stem(word) for word in text]
    return stemmed # text


def lemmatizer(text):
    """
    Applies lemmatisation to text to reduce variations of core words
    :param text: raw text string
    :return: lemmatised text string
    """
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in text]
    return lemmatized # text


def compute_TFIDF(data, max_features):
    """
    Computes the term frequency inverse document frequency feature representation for a text dataset
    :param data: pandas DataFrame with text features
    :param max_features: maximum number of TF-IDF features to extract
    :return: new DataFrame dataset with TF-IDF features 'x'
    """
    # prepare and preprocess datasets
    clean_data = prep_raw_data(data)
    # create TF-IDF vectorisation mapping over all instances
    vectoriser = TFIDF_vectorise_data(clean_data['x'], max_features)
    x_TFIDF = vectoriser.transform(clean_data['x']).toarray()
    # apply TF-IDF vectorisation to all instances of the dataset
    ind = clean_data.index
    data_TFIDF_dict = {}
    for j, i in enumerate(ind):
        data_TFIDF_dict[i] = [x_TFIDF[j, :], clean_data.loc[i]['y']]
    data_TFIDF = pd.DataFrame.from_dict(data_TFIDF_dict, orient='index', columns=['x', 'y'])
    return data_TFIDF


def TFIDF_vectorise_data(x, max_features):
    """
    Fits a TF-IDF vectoriser to input texts
    :param x: list of text features for training
    :param max_features: maximum number of features to extract
    :return: trained TF-IDF vectoriser
    """
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    vectoriser.fit(x)
    print('Number of features:', len(vectoriser.get_feature_names_out()))
    return vectoriser
