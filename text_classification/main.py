"""
-*- coding: utf-8 -*- Created on Fri 21 2020
@author: Thiago Pinho
@colaborators: Thiago Russo, Emmanuel Perotto
"""


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
from spacy.lang.pt import Portuguese
from spacy.lang.pt.stop_words import STOP_WORDS
from unidecode import unidecode
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from string import punctuation
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from preprocessing import generate_freq_dist_plot, generate_wordcloud


# ## Constants
# For better code management, the constants used in this notebook will be
# listed bellow.

VECTOR_MODEL_NAME = "pt_core_news_sm"
RELATIVE_PATH_TO_FOLDER = "./assets/datasets/ribon/"
DATA_FILENAME = "feeds_label"
NLP_SPACY = spacy.load(VECTOR_MODEL_NAME)
TARGET_VARIABLE = "LABEL_TRAIN"
POSSIBLE_TEXT_VARIABLES = ["CONTENT", "TITLE"]

# ## Load raw data and start to treat the it's structure
# We'll have a first look at the raw data and after analysing it's structure
# we can fix missing values(By dropping or artificially inserting then). We
# can encode or adjust categorical data if needed, fix column names and also
# drop unnused colummns.

"""  load the dataset """
relative_path_file = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME + ".csv"
df_ribon_news = pd.read_csv(relative_path_file)
print(df_ribon_news.info())
print()
print(df_ribon_news['Label_Train'].unique())

# ### Results
# Based on the previous step it's possible to notice two things:
#
# 1) First is that the column labels are not all uppercase or lowercase.
#
# 2) The categories avaiable to classify are not all in the same case either
#    which could lead to later confunsion on the real number of categories the
#    model should classify.
#
# So we will fix by making:
#
# 1) All **column names** will be **uppercase**
#
# 2) All **target categories** will also be **uppercase**

"""  Preprocessing the dataset names and values """
df_ribon_news.columns = map(lambda x: str(x).upper(), df_ribon_news.columns)
""" Converting all labels in TARGET_VARIABLE to uppercase """
df_ribon_news[TARGET_VARIABLE] = df_ribon_news[TARGET_VARIABLE].str.upper()
print("Column names are now: ", df_ribon_news.columns.to_list())
print()
print(
    TARGET_VARIABLE + " categories are now: ",
    df_ribon_news[TARGET_VARIABLE].unique())

# ### Storing partial progress
# One of the advantages of jupyter notebook is the possibility of only
# repeating parts of the code when there is need for it. So let's store our
# partial progress for more stability and less rework.

"""  Let"s store the data """
excel_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME + "_treated.xlsx"

"""  Convert the dataframe to an xlsx file """
df_ribon_news.to_excel(excel_filename)

print("Stored tread dataset on ", excel_filename)

# ## Load and analyse treated data
# Now we have treated some structural characteristics of the data and some
# details, let's analyse the data.

"""  Load the data for stability """
df_ribon_news_treated = pd.read_excel(excel_filename, index_col=0)
print(df_ribon_news_treated.head())

# ### Label distribution, oversampling and undersampling
# One important step is to analyse how the target categories are distributed.
# That's useful so we can better partition our data, maybe apply some over or
# undersampling if it's necessary.

"""  Let"s see how the labels are distributed """
data_labels_count = df_ribon_news_treated[TARGET_VARIABLE].value_counts()
data_labels = data_labels_count.index
average_samples_per_label = data_labels_count.mean()
standard_deviation_for_labels = data_labels_count.std()
print(
    "Mean number of samples for the target variable is: ",
    average_samples_per_label)
print(
    "Standard deviation number of samples for the target variable is: ",
    standard_deviation_for_labels)

''' Numerical analysis
    One way to analyse the frequency of certain labels is to notice with
    they're too afar from the other labels frequencies average. Let's use
    standard deviation to check it'''
def is_it_further_than_std_deviations( value ):
    is_too_much = value > average_samples_per_label + standard_deviation_for_labels
    is_too_little = value < average_samples_per_label - standard_deviation_for_labels
    if is_too_much or is_too_little:
        message = "Warning"
    else:
        message = "Okay"

    return message

for i in tqdm(range(0, len(data_labels), 2)):
    even_indexed_label = data_labels[i]
    odd_indexed_label = data_labels[i+1]

    print("{0:20}  {1:10} {2:15} {3:20} {4:10} {5:10}".format(
        even_indexed_label, data_labels_count[even_indexed_label], is_it_further_than_std_deviations(data_labels_count[even_indexed_label]),
        odd_indexed_label, data_labels_count[odd_indexed_label], is_it_further_than_std_deviations(data_labels_count[odd_indexed_label])))

''' Visual plotting'''
fig = plt.figure(figsize=(20, 8))
sns.barplot(
    x=data_labels_count.index,
    y=data_labels_count)
plt.show()

# ### Results
#
# Based on the previous step, we can see the categories **ECOLOGIA** and
# **SOLIDARIEDADE** have **more than the average added by the standard
# deviation** which can cause the model to overly recognise those labels
# patterns and make then too sensitive for those.
#
# On other hand we have the categories **FAMILIA**, **CRIANCAS** and
# **IDOSOS** with **less than the average subtracted by the standard
# deviation** which can make the model too specific for those and hardly
# classify as it.
#
# For now, let's try oversampling the least common labels by grouping then.
# When our pipeline is finely tunned we can use the grouped labels as input
# for another pipeline trainned only to discern among those.

''' One possible approach is to group up under represented labels and further
    analyse it in other pipeline.  '''
data_labels_count = df_ribon_news_treated[TARGET_VARIABLE].value_counts()
data_labels = data_labels_count.index
under_represented_labels = [
    scarse_label
    for scarse_label in tqdm(data_labels)
    if data_labels_count[scarse_label] < average_samples_per_label - standard_deviation_for_labels]
print(under_represented_labels)

''' Now we have found which ones are under represented we'll create a new
    DataFrame changing the under represented to OUTROS '''
GROUP_TARGET_LABEL = 'SCARCE_GROUP'
df_ribon_news_grouped = df_ribon_news_treated.replace({TARGET_VARIABLE: under_represented_labels}, GROUP_TARGET_LABEL)
print(df_ribon_news_grouped[TARGET_VARIABLE].value_counts())

"""  Let"s see how the labels are distributed """
data_labels_count = df_ribon_news_grouped[TARGET_VARIABLE].value_counts()
data_labels = data_labels_count.index
fig = plt.figure(figsize=(20, 8))
sns.barplot(
    x=data_labels_count.index,
    y=data_labels_count)
plt.show()

# ### Storing partial progress

excel_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME + "_treated_grouped.xlsx"

"""  Let"s store the  data """
df_ribon_news_grouped.to_excel(excel_filename)

# ## Data Partition
# Now we have treated the data structure and sampling problems. Let's drop
# unwanted columns.

"""  We then load the data for stability """
df_data = pd.read_excel(excel_filename, index_col=0)
print(df_data.head())

""" As we have two possible text_variables, let's choose one for first analysis """
text_variable = POSSIBLE_TEXT_VARIABLES[0]
""" Dropping unwanted columns """
df_data = df_data[ [text_variable] + [TARGET_VARIABLE]]
print(df_data.info())

# ### Dealing with missing values
# As there are some samples without content, they'll not be useful to train or
# to validate the model. Hapilly they're not many so let's drop them.

df_data = df_data.dropna()
print(df_data.info())

# ## Text Parsing(Preprocessing)
#
# Before we train the model, it's necessary to tokenize words, find their
# lemmas and discard some words that could mislead the model.
#
# Let's take a first look at the text variable.

raw_text_column = df_data[text_variable]
generate_wordcloud(raw_text_column)
print(generate_freq_dist_plot(raw_text_column))

# ### Symbols and stopwords
#
# As we can see, we have a lot of tokens from text variable being symbols or
# words that don't have by themselves much meaning. Let's fix that. We can
# also strip trailing spaces and remove multiple spaces.

stopwords_set = set(STOP_WORDS).union(set(stopwords.words('portuguese')))
stopword_pattern = r'\b(?:{})\b'.format(r'|'.join(stopwords_set))
symbols_pattern = '[^\w\s]'
space_pattern = r'\s{2,}'
print("This is the stopword set: ", stopword_pattern)
print()
print("This is the symbols pattern: ", symbols_pattern)
print("This is the space pattern:", space_pattern)

''' Processing text on caracteres level'''
df_data['PREPROCESSED_TEXT'] = df_data[text_variable].str.lower()
df_data['PREPROCESSED_TEXT'] = df_data['PREPROCESSED_TEXT'].str.replace(
    stopword_pattern, "")
df_data['PREPROCESSED_TEXT'] = df_data['PREPROCESSED_TEXT'].str.replace(
    symbols_pattern, "")
df_data['PREPROCESSED_TEXT'] = df_data['PREPROCESSED_TEXT'].str.replace(
    space_pattern, " ")
df_data['PREPROCESSED_TEXT'] = df_data['PREPROCESSED_TEXT'].str.strip()
generate_wordcloud(df_data['PREPROCESSED_TEXT'])
print(generate_freq_dist_plot(df_data['PREPROCESSED_TEXT']))

# ### Results
# Now the most common words are way more expressive.

# ### Lemmatizing and stemming
#

preprocessed_text_data = df_data['PREPROCESSED_TEXT'].to_list()
''' Not all variables are being undestood as strings so we have to force it'''

sentencizer = NLP_SPACY.create_pipe('sentencizer')
''' Create the pipeline 'sentencizer' component '''

try:
    ''' We then add the component to the pipeline if we hadn't done before '''
    NLP_SPACY.add_pipe(sentencizer, before='parser')
except ValueError:
    print("Pipe already present.")

print(NLP_SPACY.pipe_names)

lemmatized_doc = []
tokenized_data = []
semantics_data = []
for row in tqdm(preprocessed_text_data):
    doc = NLP_SPACY(row)
    tokenized_data.append(doc)
    semantics_data.append(" ".join([word.pos_ for word in doc]))
    lemmatized_doc.append(
        " ".join(
            [word.lemma_ if word.tag != "PRONOUN" else "" for word in doc]))

df_data['LEMMATIZED_DOC'] = lemmatized_doc

generate_wordcloud(df_data['LEMMATIZED_DOC'])
print(generate_freq_dist_plot(df_data['LEMMATIZED_DOC']))

# ### Entity Recognition
# Some parts of speech may mislead the model associating classes to certain
# entities that are not really related to the categories. The NER model(spacy
# portuguese) we are using uses the following labels:
#
# | TYPE | DESCRIPTION |
# |------|-------------------------------------------------------------------------------------------------------------------------------------------|
# | PER | Named person or family. |
# | LOC | Name of politically or geographically defined location (cities, provinces, countries, international regions, bodies of water, mountains). |
# | ORG | Named corporate, governmental, or other organizational entity. |
# | MISC | Miscellaneous entities, e.g. events, nationalities, products or works of art. |
#
# Let's take a look at the named persons or families

''' First we take a look at the found entities'''
entities_lists = []

for docs in tokenized_data:
    entities_text = ""
    for entity in docs.ents:
        if entity.label_ == "PER":
            entities_text += " " + entity.text
    entities_text = entities_text.strip()
    entities_lists.append(entities_text)
         
df_data['ENTITIES'] = entities_lists
generate_wordcloud(df_data['ENTITIES'])
print(generate_freq_dist_plot(df_data['ENTITIES']))

# ### Semantics Analysis
# Let's take a look in the parts of speech presents in the dataset

df_data['SEMANTICS'] = semantics_data
print(generate_freq_dist_plot(df_data['SEMANTICS']))


# ### Removing Entities

entities_set = set()
entities_set = set([
    word for word_list in list(map(list, df_data['ENTITIES'].str.split(" ")))
    for word in word_list])
entities_set.remove("")
entities_pattern = r'\b(?:{})\b'.format('|'.join(entities_set))

''' Processing text on entity level'''
df_data['PROCESSED_DOC'] = df_data['LEMMATIZED_DOC'].str.replace(
    entities_pattern, "")
generate_wordcloud(df_data['PROCESSED_DOC'])
print(generate_freq_dist_plot(df_data['PROCESSED_DOC']))

# ### Storing partial progress

"""  Let"s store the data """
excel_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME +\
    "_preprocessed_data.xlsx"

df_data.to_excel(excel_filename)

#  ## Text Filter(Counting and vectorizing)
#  Now we have clear tokens we can measure how much they affect the outcome
#  prediction and how many of them exist in each sample.

"""  We then load the data for stability """
df_processed_data = pd.read_excel(excel_filename, index_col=0)
print(df_processed_data.head())

''' Best parameter using GridSearch (CV score=0.535):
{'clf__alpha': 1e-05, 'clf__max_iter': 80, 'clf__penalty': 'l2',
'tfidf__norm': 'l1', 'tfidf__use_idf': True, 'vect__max_df': 0.5,
'vect__max_features': None, 'vect__ngram_range': (1, 2)}
'''
''' Text Parser
    This part is responsible to give weights to important tokens and remove
    weight for unwanted ones or those who can be misguiding.
    - Frequency Counter
    - Id-IdF Counter
'''
count_vectorizer = CountVectorizer(
    max_features=None, max_df=0.5, ngram_range=(1, 2))
tfidf_transformer = TfidfTransformer(norm='l1', use_idf='True')

''' Let's transform the lemmatized documents into count vectors '''
count_vectors = count_vectorizer.fit_transform(
    df_processed_data['PROCESSED_DOC'])

''' Then use those count vectors to generate frequency vectors '''
frequency_vectors = tfidf_transformer.fit_transform(count_vectors)

print(count_vectors[0])
print(frequency_vectors[0])

''' Model Train and Evaluation
'''

clf = SGDClassifier(alpha=1e-05, max_iter=80, penalty='l2')
pipeline_simple = Pipeline([
    ('clf', clf)
])
pipeline = Pipeline([
    ('count_vectorizer', count_vectorizer),
    ('tfidf_transformer', tfidf_transformer),
    ('clf', clf)
])

''' Let's use cross validation to better evaluate models '''
scores = cross_val_score(
    pipeline_simple,
    frequency_vectors,
    df_processed_data[TARGET_VARIABLE], cv=10)
print("Mean accuracy for explicit pipeline: ", scores.mean())

scores = cross_val_score(
    pipeline,
    df_processed_data['PROCESSED_DOC'],
    df_processed_data[TARGET_VARIABLE], cv=10)
print("Mean accuracy for implicit pipeline: ", scores.mean())

''' Let's evaluate more deeply the best model '''
X_train, X_test, y_train, y_test = train_test_split(
    df_processed_data['LEMMATIZED_DOC'],
    df_processed_data[TARGET_VARIABLE],
    test_size=0.33, random_state=42)

train1 = X_train.tolist()
labelsTrain1 = y_train.tolist()
test1 = X_test.tolist()
labelsTest1 = y_test.tolist()
"""  train """
pipeline.fit(train1, labelsTrain1)
"""  test """
preds = pipeline.predict(test1)
print("accuracy:", accuracy_score(labelsTest1, preds))
print(
    classification_report(
        labelsTest1,
        preds,
        target_names=df_processed_data[TARGET_VARIABLE].unique()))

# ### Better visualasing model classification

fig = plt.figure(figsize=(20, 20))
axes = plt.axes()

print(plot_confusion_matrix(pipeline, preds, labelsTest1, cmap='hot', ax=axes))
