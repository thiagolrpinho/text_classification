"""
-*- coding: utf-8 -*-
Created on Fri 21 2020
@author: Thiago Pinho
"""

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn import metrics
from sklearn.metrics import accuracy_score
from spacy.lang.pt import Portuguese
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import xlsxwriter
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
import string
import pyLDAvis.gensim
from tqdm.notebook import tqdm


class CleanTextTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self


def get_params(self, deep=True):
    return {}


def clean_text(text):
    text = str(text)
    text = text.strip().replace("\n", " ").replace("\r", " ")
    return text


def tokenize_text(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(
            tok.lemma_.lower().strip()
            if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    return tokens


def print_n_most_informative(vectorizer, clf, N):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)


    def lemmatizer(doc):
        """  This takes in a doc of tokens from the NER and lemmatizes them.  
            Pronouns (like "I" and "you" get lemmatized to "-PRON-",
            so I"m removing those.
        """
        doc = [token.lemma_ for token in doc if token.lemma_ != "-PRON-"]
        doc = u" ".join(doc)
        return nlp.make_doc(doc)


    def remove_stopwords(doc):
        """  This will remove stopwords and punctuation. 
            Use token.text to return strings, which we"ll need for Gensim.
        """
        doc = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
        return doc

VECTOR_MODEL_NAME = "pt_core_news_sm"
RELATIVE_PATH_TO_FOLDER = "./assets/datasets/ribon/"
DATA_FILENAME = "Feeds_Label"
parser = Portuguese()
STOPLIST = set(stopwords.words("portuguese")).add(["a", "o", "A", "O"])
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]
TARGET_VARIABLE = "LABEL_TRAIN"
TEXT_VARIABLE = "TITLE"
print(STOPLIST)
print(SYMBOLS)

"""  load the dataset """
relative_path_file = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME + ".csv"
df_ribon_news = pd.read_csv(relative_path_file)
print(df_ribon_news.head())

"""  Preprocess the dataset names and values """
df_ribon_news.columns = map(lambda x: str(x).upper(), df_ribon_news.columns)
df_ribon_news[TARGET_VARIABLE] = df_ribon_news[TARGET_VARIABLE].str.upper()
""" Converting all labels to lowercase """
print(df_ribon_news.head())

"""  Let"s see how the labels are distributed """
fig = plt.figure(figsize=(16, 8))
sns.barplot(
    x=df_ribon_news[TARGET_VARIABLE].unique(),
    y=df_ribon_news[TARGET_VARIABLE].value_counts())
plt.show()

"""  Let"s store the data """
excel_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME + "_treated.xlsx"

"""  Convert the dataframe to an xlsx file """
df_ribon_news.to_excel(excel_filename)

"""  We then load the data for stability """
df_ribon_news_treated = pd.read_excel(excel_filename, index_col=0)
print(df_ribon_news_treated.head())

for label in df_ribon_news_treated[TARGET_VARIABLE].unique():
    print(
        label + ": ",
        len(df_ribon_news_treated[
            df_ribon_news_treated[TARGET_VARIABLE] == label]))

""" As we have two text variables CONTENT and TITLE.
    we can use both of then to improve predictions
"""
first_pipeline_text_variable = "CONTENT"
second_pipeline_text_variable = "TITLE"

df_first_ribon_news_data = df_ribon_news_treated[
    [first_pipeline_text_variable, TARGET_VARIABLE]]
df_second_ribon_news_data = df_ribon_news_treated[
    [second_pipeline_text_variable, TARGET_VARIABLE]]

"""  Let"s store the data """
excel_first_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME +\
    "_first_data.xlsx"
excel_second_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME +\
    "_second_data.xlsx"

df_first_ribon_news_data.to_excel(excel_first_filename)
df_second_ribon_news_data.to_excel(excel_second_filename)

"""  We then load the data for stability """
df_first_data = pd.read_excel(excel_first_filename, index_col=0)
df_second_data = pd.read_excel(excel_second_filename, index_col=0)
print(df_first_data.head())
print(df_second_data.head())

"""  Let"s create a train and test sample """
"""  for TITLE -> TRAIN_LABEL analysis """
y_variable = "LABEL_TRAIN"
X_variable = first_pipeline_text_variable
df_X_data = df_first_data[X_variable]
df_y_data = df_first_data[y_variable]

X_train, X_test, y_train, y_test = train_test_split(
    df_X_data.values,
    df_y_data.values,
    test_size=0.33, random_state=42)
print("One sample of the " + X_variable + " column: ", X_train[0])
print("This sample respective " + y_variable + "column: ", y_train[0])
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", y_train.shape)

vectorizer = CountVectorizer(tokenizer=tokenize_text, ngram_range=(1, 1))
clf = LinearSVC()

pipe = Pipeline(
    [("cleanText", CleanTextTransformer()),
        ("vectorizer", vectorizer), ("clf", clf)])
"""  data """
train1 = X_train.tolist()
labelsTrain1 = y_train.tolist()
test1 = X_test.tolist()
labelsTest1 = y_test.tolist()
"""  train """
pipe.fit(train1, labelsTrain1)
"""  test """
preds = pipe.predict(test1)
print("accuracy:", accuracy_score(labelsTest1, preds))
print("Top 10 features used to predict: ")
print_n_most_informative(vectorizer, clf, 10)

print(
    metrics.classification_report(
        labelsTest1,
        preds,
        target_names=df_first_data[y_variable].unique()))

"""  Let"s create a train and test sample
    for TITLE -> TRAIN_LABEL analysis
"""
y_variable = "LABEL_TRAIN"
X_variable = second_pipeline_text_variable
df_X_data = df_second_data[X_variable]
df_y_data = df_second_data[y_variable]

X_train, X_test, y_train, y_test = train_test_split(
    df_X_data.values,
    df_y_data.values,
    test_size=0.33, random_state=42)
print("One sample of the " + X_variable + " column: ", X_train[0])
print("This sample respective " + y_variable + "column: ", y_train[0])
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", y_train.shape)

vectorizer = CountVectorizer(tokenizer=tokenize_text, ngram_range=(1, 1))
tf_transformer = TfidfTransformer(use_idf=False)
clf = LinearSVC()

pipe = Pipeline(
    [("clean_text", CleanTextTransformer()),
        ("vectorizer", vectorizer),
        ("tfidf", tf_transformer)
        ("clf", clf)])
"""  data """
train1 = X_train.tolist()
labelsTrain1 = y_train.tolist()
test1 = X_test.tolist()
labelsTest1 = y_test.tolist()
"""  train """
pipe.fit(train1, labelsTrain1)
"""  test """
preds = pipe.predict(test1)
print("accuracy:", accuracy_score(labelsTest1, preds))
print("Top 10 features used to predict: ")
print_n_most_informative(vectorizer, clf, 10)

print(
    metrics.classification_report(
        labelsTest1,
        preds,
        target_names=df_second_data[y_variable].unique()))

"""  Now we found that which text variable is more descriptive let's decide which model
     have a better perfomance with this kind of data 
"""
y_variable = "LABEL_TRAIN"
X_variable = first_pipeline_text_variable
df_X_data = df_first_data[X_variable]
df_y_data = df_first_data[y_variable]

vectorizer = CountVectorizer(tokenizer=tokenize_text, ngram_range=(1, 1))
tf_transformer = TfidfTransformer(use_idf=True)
clf = CalibratedClassifierCV()

first_pipe = Pipeline(
    [("clean_text", CleanTextTransformer()),
        ("vectorizer", vectorizer),
        ("tfidf", tf_transformer),
        ("clf", clf)])

scores = cross_val_score(first_pipe, df_X_data, df_y_data, cv=5)
print("Mean model scores for first:", scores.mean())

svm = SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=10, tol=None)

second_pipe = Pipeline(
    [("clean_text", CleanTextTransformer()),
        ("vectorizer", vectorizer),
        ("tfidf", tf_transformer),
        ("svm", svm)])

scores = cross_val_score(second_pipe, df_X_data, df_y_data, cv=5)
print("Mean model scores for first:", scores.mean())


