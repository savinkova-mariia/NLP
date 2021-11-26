import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def process_review(review):

    stemmer = PorterStemmer()
    stopwords_rus = stopwords.words('russian')
    # remove stock market tickers like $GE
    review = re.sub(r'\$\w*', '', review)
    # remove old style retweet text "RT"
    review = re.sub(r'^RT[\s]+', '', review)
    # remove hyperlinks
    review = re.sub(r'https?:\/\/.*[\r\n]*', '', review)
    # remove hashtags
    # only removing the hash # sign from the word
    review = re.sub(r'#', '', review)

    tokenizer = word_tokenize(text, language='russian')
    review_tokens = tokenizer.tokenize(review)

    reviews_clean = []
    for word in review_tokens:
        if (word not in stopwords_rus and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            stem_word = stemmer.stem(word)  # stemming word
            review_clean.append(stem_word)

    return reviews_clean


def build_freqs(reviews, ys):
    yslist = np.squeeze(ys).tolist()


    freqs = {}
    for y, review in zip(yslist, reviews):
        for word in process_review(review):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs
