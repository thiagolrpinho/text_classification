# -*- coding: utf-8 -*-
"""
Created on Fri 21 2020
@author: Thiago Pinho
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import spacy


VECTOR_MODEL_NAME = 'pt_core_news_sm'
RELATIVE_PATH_TO_CSV = "./assets/datasets/ribon/Feeds_Label.csv"


# load the dataset
df_ribon_news = pd.read_csv(RELATIVE_PATH_TO_CSV)
print(df_ribon_news.head())


# Preprocess the dataset names and values
df_ribon_news.columns = map(lambda x: str(x).upper(), df_ribon_news.columns)
df_ribon_news['LABEL_TRAIN'] = df_ribon_news['LABEL_TRAIN'].str.upper()
''' Converting all labels to lowercase '''
print(df_ribon_news.head())


# Viewing frequencies
for label in df_ribon_news['LABEL_TRAIN'].unique():
    print(label + ": ", len(df_ribon_news[df_ribon_news.LABEL_TRAIN == label]))

# For simple label encoding
# All categorical columns
categorical_cols = ['LABEL_TRAIN']

label_df_ribon_news = df_ribon_news

# Apply label encoder
label_encoder = LabelEncoder()
for col in categorical_cols:
    label_df_ribon_news[col] = label_encoder.fit_transform(df_ribon_news[col])

print(label_df_ribon_news.head())

# Load the large model to get the vectors
nlp = spacy.load(VECTOR_MODEL_NAME)

# We just want the vectors so we can turn off other models in the pipeline
with nlp.disable_pipes():
    vectors = np.array(
        [nlp(str(news.CONTENT)).vector
            for idx, news in label_df_ribon_news.iterrows()])

print(vectors.shape)

# Training models
X_train, X_test, y_train, y_test = train_test_split(vectors, label_df_ribon_news.LABEL_TRAIN, 
                                                    test_size=0.1, random_state=1)

# Create the LinearSVC model
model = LinearSVC(random_state=1, dual=False)
#Fit the model
model.fit(X_train, y_train)

# Uncomment and run to see model accuracy
print(f'Model test accuracy: {model.score(X_test, y_test)*100:.3f}%')

# Scratch space in case you want to experiment with other models

second_model = RandomForestRegressor(random_state=2)
second_model.fit(X_train, y_train)
print(f'Model test accuracy: {second_model.score(X_test, y_test)*100:.3f}%')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import string
import re
from collections import Counter
from nltk.corpus import stopwords
stopwords = stopwords.words('portuguese')
print(stopwords)

TARGET_VARIABLE = 'LABEL_TRAIN'
TEXT_VARIABLE = 'TITLE'

from sklearn.model_selection import train_test_split
print(df_ribon_news.columns)
train, test = train_test_split(df_ribon_news[[TARGET_VARIABLE, TEXT_VARIABLE]], 
                                                    test_size=0.33,
                                                        random_state=42)
print('Research title sample:', train[TEXT_VARIABLE].iloc[0])
print('Conference of this paper:', train[TARGET_VARIABLE].iloc[0])
print('Training Data Shape:', train.shape)
print('Testing Data Shape:', test.shape)


fig = plt.figure(figsize=(8,4))
sns.barplot(x = train[TARGET_VARIABLE].unique(), y=train[TARGET_VARIABLE].value_counts())
plt.show()

punctuations = string.punctuation
print(punctuations)

def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


most_common_target_label = train[TARGET_VARIABLE].value_counts().idxmax()

most_common_target_text = [text for text in train[train[TARGET_VARIABLE] == most_common_target_label][TEXT_VARIABLE]]
most_common_target_clean = cleanup_text(most_common_target_text)
most_common_target_clean = ' '.join(most_common_target_clean).split()
most_common_target_counts = Counter(most_common_target_clean)
most_common_target_common_words = [word[0] for word in most_common_target_counts.most_common(20)]
most_common_target_common_counts = [word[1] for word in most_common_target_counts.most_common(20)]
fig = plt.figure(figsize=(18,6))
sns.barplot(x=most_common_target_common_words, y=most_common_target_common_counts)
plt.title('Most Common Words used in ' + TEXT_VARIABLE + ' of ' + most_common_target_label)
plt.show()


vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
clf = LinearSVC()

pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
# data
train1 = train[TEXT_VARIABLE].tolist()
labelsTrain1 = train[TARGET_VARIABLE].tolist()
test1 = test[TEXT_VARIABLE].tolist()
labelsTest1 = test[TARGET_VARIABLE].tolist()
# train
pipe.fit(train1, labelsTrain1)
# test
preds = pipe.predict(test1)
print("accuracy:", accuracy_score(labelsTest1, preds))
print("Top 10 features used to predict: ")
printNMostInformative(vectorizer, clf, 10)

pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer)])
transform = pipe.fit_transform(train1, labelsTrain1)
vocab = vectorizer.get_feature_names()
for i in range(len(train1)):
    s = ""
    indexIntoVocab = transform.indices[transform.indptr[i]:transform.indptr[i+1]]
    numOccurences = transform.data[transform.indptr[i]:transform.indptr[i+1]]
    for idx, num in zip(indexIntoVocab, numOccurences):
        s += str((vocab[idx], num))
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
clf = LinearSVC()

pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
# data
train1 = train[TEXT_VARIABLE].tolist()
labelsTrain1 = train[TARGET_VARIABLE].tolist()
test1 = test[TEXT_VARIABLE].tolist()
labelsTest1 = test[TARGET_VARIABLE].tolist()
# train
pipe.fit(train1, labelsTrain1)
# test
preds = pipe.predict(test1)
print("accuracy:", accuracy_score(labelsTest1, preds))
print("Top 10 features used to predict: ")
printNMostInformative(vectorizer, clf, 10)

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
clf = LinearSVC()

pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
# data
train1 = train[TEXT_VARIABLE].tolist()
labelsTrain1 = train[TARGET_VARIABLE].tolist()
test1 = test[TEXT_VARIABLE].tolist()
labelsTest1 = test[TARGET_VARIABLE].tolist()
# train
pipe.fit(train1, labelsTrain1)
# test
preds = pipe.predict(test1)
print("accuracy:", accuracy_score(labelsTest1, preds))
print("Top 10 features used to predict: ")
printNMostInformative(vectorizer, clf, 10)

from sklearn import metrics
print(metrics.classification_report(labelsTest1, preds, 
                                    target_names=df[TARGET_VARIABLE].unique()))

TARGET_VARIABLE = 'LABEL_TRAIN'
TEXT_VARIABLE = 'CONTENT'

from sklearn.model_selection import train_test_split
print(df_ribon_news.columns)
train, test = train_test_split(df_ribon_news[[TARGET_VARIABLE, TEXT_VARIABLE]], 
                                                      test_size=0.33,
                                                          random_state=42)
print('Research title sample:', train[TEXT_VARIABLE].iloc[0])
print('Conference of this paper:', train[TARGET_VARIABLE].iloc[0])
print('Training Data Shape:', train.shape)
print('Testing Data Shape:', test.shape)

most_common_target_label = train[TARGET_VARIABLE].value_counts().idxmax()
print(most_common_target_label)
most_common_target_text = [str(text) for text in train[train[TARGET_VARIABLE] == most_common_target_label][TEXT_VARIABLE]]
most_common_target_clean = cleanup_text(most_common_target_text)
most_common_target_clean = ' '.join(most_common_target_clean).split()
most_common_target_counts = Counter(most_common_target_clean)
most_common_target_common_words = [word[0] for word in most_common_target_counts.most_common(20)]
most_common_target_common_counts = [word[1] for word in most_common_target_counts.most_common(20)]
fig = plt.figure(figsize=(18,6))
sns.barplot(x=most_common_target_common_words, y=most_common_target_common_counts)
plt.title('Most Common Words used in ' + TEXT_VARIABLE + ' of ' + most_common_target_label)
plt.show()

pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer)])
transform = pipe.fit_transform(train1, labelsTrain1)
vocab = vectorizer.get_feature_names()
for i in range(len(train1)):
    s = ""
    indexIntoVocab = transform.indices[transform.indptr[i]:transform.indptr[i+1]]
    numOccurences = transform.data[transform.indptr[i]:transform.indptr[i+1]]
    for idx, num in zip(indexIntoVocab, numOccurences):
        s += str((vocab[idx], num))
        
from sklearn import metrics
print(metrics.classification_report(labelsTest1, preds, 
                                    target_names=df_ribon_news[TARGET_VARIABLE].unique()))


import numpy as np
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_lg

from tqdm import tqdm_notebook as tqdm
from pprint import pprint

doc_list = []
# Iterates through each article in the corpus.
for doc in tqdm(newest_doc):
    # Passes that article through the pipeline and adds to a new list.
    pr = nlp(doc)
    doc_list.append(pr)

def lemmatizer(doc):
    # This takes in a doc of tokens from the NER and lemmatizes them.
    # Pronouns (like "I" and "you" get lemmatized to '-PRON-',
    # so I'm removing those.
    doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    doc = u' '.join(doc)
    return nlp.make_doc(doc)
    
def remove_stopwords(doc):
    # This will remove stopwords and punctuation.
    # Use token.text to return strings, which we'll need for Gensim.
    doc = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    return doc

# The add_pipe function appends our functions to the default pipeline.
nlp.add_pipe(lemmatizer,name='lemmatizer',after='ner')
nlp.add_pipe(remove_stopwords, name="stopwords", last=True)


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=words,
                                        num_topics=10, random_state=2,
                                        update_every=1, passes=10,
                                        alpha='auto', per_word_topics=True)

import pyLDAvis.gensim
# turn on automatic rendering of visualizations
pyLDAvis.enable_notebook()

pyLDAvis.gensim.prepare(lda_model, corpus, words)


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['come','order','try','go','get','make','drink','plate','dish','restaurant','place',
                  'would','really','like','great','service','came','got'])

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count = bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod
    
def get_corpus(df, text_label, target_label):
    df[text_label] = str(df.text_label)
    words = list(sent_to_words(df.text_label))
    words = remove_stopwords(words)
    bigram_mod = bigrams(words)
    bigram = [bigram_mod[word] for word in words]
    id2word = gensim.corpora.Dictionary(bigram)
    id2word.filter_extremes(no_below=10, no_above=0.35)
    id2word.compactify()
    corpus = [id2word.doc2bow(text_label) for text_label in bigram]
    return corpus, id2word, bigram

train_corpus, train_id2word, bigram_train = get_corpus(df_ribon_news)

