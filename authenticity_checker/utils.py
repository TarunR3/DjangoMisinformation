import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from string import punctuation
import re
from bs4 import BeautifulSoup

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    def __init__(self):
        self.stopwords_set = set(stopwords.words('english'))
        self.stopwords_set.update(list(punctuation))
    
    def extract_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def remove_url(self, text):
        return re.sub(r'http\S+', '', text)

    def remove_noise(self, text):
        # Remove stopwords, punctuation, convert all text to lower case, and remove double spaces
        text = self.extract_html(text)
        text = self.remove_url(text)
        ret_text = []
        for word in re.split(r'\s+', text):
            if word.lower() not in self.stopwords_set:
                ret_text.append(word.lower())
        
        return " ".join(ret_text)

class AdvancedTextPreprocessor:
    def __init__(self):
        self.stopwords_set = set(stopwords.words('english'))
        self.stopwords_set.update(list(punctuation))
        self.lemmatizer = WordNetLemmatizer()

    def extract_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def remove_url(self, text):
        return re.sub(r'http\S+|www\.\S+', '', text)

    def lemmatize_text(self, text):
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in word_tokenize(text)]
        return " ".join(lemmatized_words)

    def remove_noise(self, text):
        text = self.extract_html(text)
        text = self.remove_url(text)
        ret_text = []
        for word in re.split(r'\s+', text):
            word = word.lower()
            if word not in self.stopwords_set and not word.isdigit():
                ret_text.append(word)
        
        clean_text = " ".join(ret_text)
        clean_text = self.lemmatize_text(clean_text)
        return re.sub(r'\s+', ' ', clean_text).strip()
