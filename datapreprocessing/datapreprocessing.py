import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import contractions

stop_words = stopwords.words('english')
new_stopwords = ["mario","la","blah","saturday","monday","sunday","morning","evening","friday","would","shall","could","might"]
stop_words.extend(new_stopwords)
stop_words.remove("not")
stop_words=set(stop_words)

# Removing special character
def remove_special_character(content):
    return re.sub('\W+',' ', content )#re.sub('\[[^&@#!]]*\]', '', content)

# Removing Url's
def remove_url(content):
    return re.sub(r'http\S+', '', content)

# Removing the stopwords from text
def remove_stopwords(content):
    clean_data = []
    for word in content.split():
        if word.strip().lower() not in stop_words and word.strip().lower().isalpha():
            clean_data.append(word.strip().lower())
    return " ".join(clean_data)

def contraction_expansion(content):
    return contractions.fix(content)

def data_cleaning(content):
    content = contraction_expansion(content)
    content = remove_special_character(content)
    content = remove_url(content)
    content = remove_stopwords(content)
    return content

class DataCleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("calling --init--")
    def fit(self, X, y=None):
        print("Calling fit")
        return self
    def transform(self, X, y=None):
        print("Calling transform")
        X = X.apply(data_cleaning)
        return X

# Lemmatization of word
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()
    def __call__(self, reviews):
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]
