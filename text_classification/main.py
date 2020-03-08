#!/usr/bin/env python
# coding: utf-8


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
# For better code management, the constants used in this notebook will be listed bellow.


VECTOR_MODEL_NAME = "pt_core_news_sm"
RELATIVE_PATH_TO_FOLDER = "./assets/datasets/ribon/"
DATA_FILENAME = "feeds_label"
NLP_SPACY = spacy.load(VECTOR_MODEL_NAME)
TARGET_VARIABLE = "LABEL_TRAIN"
POSSIBLE_TEXT_VARIABLES = ["CONTENT", "TITLE"]


# ## Load raw data and start to treat the it's structure
# We'll have a first look at the raw data and after analysing it's structure we can fix missing values(By dropping or artificially inserting then). We can encode or adjust categorical data if needed, fix column names and also drop unnused colummns.


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
# 2) The categories avaiable to classify are not all in the same case either which could lead to later confunsion on the real number of categories the model should classify.
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
print(TARGET_VARIABLE + " categories are now: ", df_ribon_news[TARGET_VARIABLE].unique())


# ### Storing partial progress
# One of the advantages of jupyter notebook is the possibility of only repeating parts of the code when there is need for it. So let's store our partial progress for more stability and less rework.


"""  Let"s store the data """
excel_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME + "_treated.xlsx"



"""  Convert the dataframe to an xlsx file """
df_ribon_news.to_excel(excel_filename)

print("Stored tread dataset on ", excel_filename)


# ## Load and analyse treated data
# Now we have treated some structural characteristics of the data and some details, let's analyse the data.


"""  Load the data for stability """
df_ribon_news_treated = pd.read_excel(excel_filename, index_col=0)
print(df_ribon_news_treated.info())


# ### Choosing text data and dropping unwanted variables
# Not all columns available in data will be useful for the label classification.


""" In the previous results, we could that are two text variables besides the target: CONTENT and TITLE.
There's also the numeric variable pick_count which is unrelated to label, so let's add it to a unwanted list """
unwanted_columns = set(['PICK_COUNT', 'ID'])

""" As CONTENT is empty in two cases let's compare it to title which is not empty in any case """
compared_columns = set(['CONTENT', 'TITLE'])
columns_stats = []
columns_series = []
for column in compared_columns:
    column_series = df_ribon_news_treated[column]
    columns_stats.append((column_series.str.len().mean(), column_series.str.len().std()))
    columns_series.append(column_series)

for column, stats in zip(compared_columns, columns_stats):
    mean, std = stats
    mean = str(int(mean))
    std = str(int(std))
    print(
        "Column " + column + " mean length was " + mean + " and standard deviation was " + std)


# ### Results
# As CONTENT is appears to have more data, it could bring better results. But as two rows have this column empty we would have to drop those. One way around it is to oversample the data by using both as text variables.


unwanted_columns.union(compared_columns)
wanted_columns = set(df_ribon_news_treated.columns).intersection(unwanted_columns)
df_preprocessed_data = pd.DataFrame(columns=[TARGET_VARIABLE, "TEXT_VARIABLE"])
for column in compared_columns:
    df_labels_texts_variables = df_ribon_news_treated[[TARGET_VARIABLE, column]]
    df_labels_texts_variables = df_labels_texts_variables.rename(columns={column:"TEXT_VARIABLE"})
    df_preprocessed_data = df_preprocessed_data.append(df_labels_texts_variables, ignore_index=True)

print(df_preprocessed_data.info())


# ### Dealing with missing values
# As there are some samples that are empty, they'll not be useful to train or to validate the model. 
# Let's drop them


df_preprocessed_data = df_preprocessed_data.dropna()
print(df_preprocessed_data.info())


# ### Label distribution, oversampling and undersampling
# One important step is to analyse how the target categories are distributed. That's useful so we can better partition our data, maybe apply some over or undersampling if it's necessary.


"""  Let"s see how the labels are distributed """
data_labels_count = df_preprocessed_data[TARGET_VARIABLE].value_counts()
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
# Based on the previous step, we can see the categories **ECOLOGIA** and **SOLIDARIEDADE** have **more than the average added by the standard deviation** which can cause the model to overly recognise those labels patterns and make then too sensitive for those. 
# 
# On other hand we have the categories **FAMILIA**, **CRIANCAS** and **IDOSOS** with **less than the average subtracted by the standard deviation** which can make the model too specific for those and hardly classify as it.
# 
# Let's oversample the least common labels by grouping then. When our pipeline is finely tunned we can use the grouped labels as input for another pipeline trainned only to discern among those.
# And also undersample the most common labels by ramdonly select less samples.


''' Let's create another dataframe and find which samples will be and how they'll be part of it'''
data_labels_count = df_preprocessed_data[TARGET_VARIABLE].value_counts()
data_labels = data_labels_count.index
under_represented_labels = []
over_represented_labels = []
max_number_of_samples = average_samples_per_label + standard_deviation_for_labels
min_number_of_samples = average_samples_per_label - standard_deviation_for_labels
for label in tqdm(data_labels):
    if data_labels_count[label] < min_number_of_samples:
        under_represented_labels.append(label)
    elif data_labels_count[label] > max_number_of_samples:
        over_represented_labels.append(label)

unchanged_labels = list(set(data_labels) - set(under_represented_labels) - set(over_represented_labels))

print("Let's check the labels found: ")
print("Underpresented labels: ", under_represented_labels)
print("Overrepresented labels: ", over_represented_labels)
print("Unchanged Labels: ", unchanged_labels)

df_preprocessed_grouped = pd.DataFrame(columns=df_preprocessed_data.columns)

for label in unchanged_labels:
    unchanged_rows = df_preprocessed_data[df_preprocessed_data[TARGET_VARIABLE] == label]
    df_preprocessed_grouped = df_preprocessed_grouped.append(unchanged_rows)

''' Now we have found which ones are under represented we'll add them to the new
    DataFrame and then change the under represented label to SCARCE_GROUP '''
for label in under_represented_labels:
    under_represented_rows = df_preprocessed_data[df_preprocessed_data[TARGET_VARIABLE] == label]
    df_preprocessed_grouped = df_preprocessed_grouped.append(under_represented_rows)

GROUP_TARGET_LABEL = 'SCARCE_GROUP'
df_preprocessed_grouped = df_preprocessed_grouped.replace(
    {TARGET_VARIABLE: under_represented_labels}, GROUP_TARGET_LABEL)

""" For the over represented, we'll select some of the samples."""
for label in over_represented_labels:
    over_represented_rows = df_preprocessed_data[
        df_preprocessed_data[TARGET_VARIABLE] == label].sample(int(max_number_of_samples))
    df_preprocessed_grouped = df_preprocessed_grouped.append(over_represented_rows)

print(df_preprocessed_grouped[TARGET_VARIABLE].value_counts())

"""  Let"s see how the labels are distributed """
data_labels_count = df_preprocessed_grouped[TARGET_VARIABLE].value_counts()
data_labels = data_labels_count.index
fig = plt.figure(figsize=(20, 8))
sns.barplot(
    x=data_labels_count.index,
    y=data_labels_count)
plt.show()
print(df_preprocessed_data.info())


# ### Storing partial progress


excel_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME +    "_treated_grouped.xlsx"



"""  Let"s store the  data """
df_preprocessed_grouped.to_excel(excel_filename)


# ## Data Partition
# Now we have treated the data structure and sampling problems. Let's drop unwanted columns.


"""  We then load the data for stability """
df_data = pd.read_excel(excel_filename, index_col=0)
print(df_data.head())


# ## Text Filter(Preprocessing)
# 
# Before we train the model, it's necessary to tokenize words, find their lemmas and discard some words that could mislead the model.
# 
# Let's take a first look at the text variable.


text_variable = 'TEXT_VARIABLE'
raw_text_column = df_data[text_variable]
generate_wordcloud(raw_text_column)
print(generate_freq_dist_plot(raw_text_column))


# ### Symbols and stopwords
# 
# As we can see, we have a lot of tokens from text variable being symbols or words that don't have by themselves much meaning. Let's fix that.
# We can also strip trailing spaces and remove multiple spaces.


stopwords_set = set(STOP_WORDS).union(set(stopwords.words('portuguese'))).union(set(['anos', 'ano', 'dia', 'dias']))
stopword_pattern = r'\b(?:{})\b'.format(r'|'.join(stopwords_set))
symbols_pattern = r'(?:[{}]|[^\w\s])'.format(punctuation)
space_pattern = r'\s\s+'
number_pattern = r'\d'
print("This is the stopword list: ", sorted(list(stopwords_set)))
print("This is the number pattern:", number_pattern)
print("This is the symbols pattern: ", symbols_pattern)
print("This is the space pattern:", space_pattern)



''' Processing text on caracteres level'''
df_data['PREPROCESSED_TEXT'] = df_data[text_variable]
df_data['PREPROCESSED_TEXT'] = df_data['PREPROCESSED_TEXT'].str.replace(
    number_pattern, " ")
df_data['PREPROCESSED_TEXT'] = df_data['PREPROCESSED_TEXT'].str.replace(
    stopword_pattern, " ", case=False)
df_data['PREPROCESSED_TEXT'] = df_data['PREPROCESSED_TEXT'].str.replace(
    symbols_pattern, " ")
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

tokenized_data = []
semantics_data = []
lemmatized_doc = []
normalized_doc = []
raw_doc = []
for row in tqdm(preprocessed_text_data):
    doc = NLP_SPACY(row)
    tokenized_data.append(doc)
    raw_doc.append(
        " ".join(
            [word.text for word in doc]))
    lemmatized_doc.append(
        " ".join(
            [word.lemma_ for word in doc]))
    normalized_doc.append(
        " ".join(
            [word.norm_ for word in doc]))
    
df_data['RAW_DOC'] = raw_doc
df_data['NORMALIZED_DOC'] = normalized_doc
df_data['LEMMATIZED_DOC'] = lemmatized_doc

print("Documents without lemmatization")
print(generate_freq_dist_plot(df_data['RAW_DOC']))
generate_wordcloud(df_data['RAW_DOC'])
print("Documents with minor lemmatization")
print(generate_freq_dist_plot(df_data['NORMALIZED_DOC']))
generate_wordcloud(df_data['NORMALIZED_DOC'])
print("Documents with full lemmatization")
print(generate_freq_dist_plot(df_data['LEMMATIZED_DOC']))
generate_wordcloud(df_data['LEMMATIZED_DOC'])


# ### Entity Recognition
# Some parts of speech may mislead the model associating classes to certain entities that are not really related to the categories.
# The NER model(spacy portuguese) we are using uses the following labels:
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
entity_unwanted_types = set(['PER', 'ORG'])

for docs in tokenized_data:
    entities_text = ""
    for entity in docs.ents:
        if entity.label_ in entity_unwanted_types:
            entities_text += " " + entity.text
    entities_text = entities_text.strip()
    entities_lists.append(entities_text)
            
df_data['ENTITIES'] = entities_lists
generate_wordcloud(df_data['ENTITIES'])
print(generate_freq_dist_plot(df_data['ENTITIES']))


# ### Removing Entities


entities_set = set()
entities_set = set([ word for word_list in list(map(list, df_data['ENTITIES'].str.split(" ")))
                            for word in word_list ])
entities_set.remove("")
entities_pattern = r'\b(?:{})\b'.format('|'.join(entities_set)) 

''' Processing text on entity level'''
df_data['PROCESSED_DOC'] = df_data['NORMALIZED_DOC'].str.replace(entities_pattern, " ")
generate_wordcloud(df_data['PROCESSED_DOC'])
print(generate_freq_dist_plot(df_data['PROCESSED_DOC']))


# ### POS Analysis
# Let's take a look in the parts of speech presents in the dataset


semantics_data = []
for doc in tokenized_data:
    semantics_data.append(" ".join([word.pos_ for word in doc]))

df_data['SEMANTICS'] = semantics_data
print(generate_freq_dist_plot(df_data['SEMANTICS']))



ALLOWED_POS = set(["PROPN", "NOUN", "ADV", "ADJ", "VERB"])

unwanted_pos_text = []
for doc in tokenized_data:
    unwanted_pos_text.append(
        " ".join(
            [word.lemma_ if not str(word.pos_) in ALLOWED_POS else "" for word in doc]))
    
df_data['UNWANTED_POS'] = unwanted_pos_text
generate_wordcloud(df_data['UNWANTED_POS'])
print(generate_freq_dist_plot(df_data['UNWANTED_POS']))


# ### Removing POS


ALLOWED_POS = set(["PROPN", "NOUN", "ADV", "ADJ", "VERB"])

processed_doc = []
for doc in tokenized_data:
    processed_doc.append(
        " ".join(
            [word.norm_ if str(word.pos_) in ALLOWED_POS else "" for word in doc]))

df_data['PROCESSED_DOC'] = processed_doc
''' Processing text on entity level again '''
df_data['PROCESSED_DOC'] = df_data['PROCESSED_DOC'].str.replace(entities_pattern, " ")
generate_wordcloud(df_data['PROCESSED_DOC'])
print(generate_freq_dist_plot(df_data['PROCESSED_DOC']))



""" Removing extra spaces originated from processing """
df_data['PROCESSED_DOC'] = df_data['PROCESSED_DOC'].str.replace(space_pattern, " ").str.strip()
df_data['UNWANTED_POS'] = df_data['UNWANTED_POS'].str.replace(space_pattern, " ").str.strip()


# ### Viewing the most common words for each label


target_labels = df_data[TARGET_VARIABLE].unique()

for label in target_labels:
    words_for_label = df_data[df_data[TARGET_VARIABLE] == label]
    print("Label: ", label)
    print(generate_freq_dist_plot(words_for_label['PROCESSED_DOC']))
    generate_wordcloud(df_data['PROCESSED_DOC'])


# ### Storing partial progress


"""  Let"s store the data """
excel_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME +    "_processed_data.xlsx"



df_data.to_excel(excel_filename)


#  ## Text Parser(Counting and vectorizing)
#  Now we have clear tokens we can measure how much they affect the outcome prediction and how many of them exist in each sample.


"""  We then load the data for stability """
df_processed_data = pd.read_excel(excel_filename, index_col=0)
print(df_processed_data.info())


# ### Dealing with missing values
# As there are some samples without content, they'll not be useful to train or to validate the model. 
# Hapilly they're not many so let's drop them.


missing_variables = ['ENTITIES', 'UNWANTED_POS']
df_processed_data = df_processed_data.drop(columns=missing_variables).dropna()
print(df_processed_data.info())


# ### Choosing best parameters for Counting and Vectorizing


is_gridsearching = False
if is_gridsearching:
    from sklearn.metrics import make_scorer
    from sklearn.metrics import accuracy_score

    search_count_vectorizer = CountVectorizer()
    search_tfidf_transformer = TfidfTransformer()
    clf = SGDClassifier(alpha=1e-05, max_iter=80, penalty='l2')

    search_params = {
        'vect__min_df': np.arange(0, 0.001, 0.0003),
        'vect__max_df': np.arange(0.2, 0.9, 0.3),
        'vect__max_features': [None],
        'vect__ngram_range': [(1, 2), (1, 3), (2, 3)],
        'tfidf__norm': ['l2'],
        'tfidf__use_idf': [False, True],
        'tfidf__smooth_idf': [False],
        'tfidf__sublinear_tf' : [False, True]}

    search_pipeline = Pipeline([
        ('vect', search_count_vectorizer),
        ('tfidf', search_tfidf_transformer),
        ('clf', clf)
    ])

    gs = GridSearchCV(search_pipeline,
                      param_grid=search_params, cv=5)
    gs.fit(df_data['PROCESSED_DOC'].values, df_data[TARGET_VARIABLE])
    results = gs.cv_results_
    print(results)



print(gs.best_params_)



''' Best parameter using GridSearch (CV score=0.535):
{'clf__alpha': 1e-05, 'clf__max_iter': 80, 'clf__penalty': 'l2', 'tfidf__norm': 'l1',
'tfidf__smooth_idf': False, 'tfidf__sublinear_tf': True, 'tfidf__use_idf': True,
'vect__max_df': 0.6000000000000001, 'vect__max_features': None, 'vect__min_df': 0.0007,
'vect__ngram_range': (1, 2)}
'''
''' Text Parser
    This part is responsible to give weights to important tokens and remove
    weight for unwanted ones or those who can be misguiding.
    - Frequency Counter
    - Id-IdF Counter
'''
count_vectorizer = CountVectorizer(
    max_features=None, min_df=0.0007, max_df=0.6, ngram_range=(1, 2))
tfidf_transformer = TfidfTransformer(norm='l1', use_idf=True, sublinear_tf=True)

''' Let's transform the lemmatized documents into count vectors '''
count_vectors = count_vectorizer.fit_transform(
    df_processed_data['PROCESSED_DOC'])

''' Then use those count vectors to generate frequency vectors '''
frequency_vectors = tfidf_transformer.fit_transform(count_vectors)

print(count_vectors[0])
print(frequency_vectors[0])



''' Let's transform the lemmatized documents into count vectors '''
count_vectorizer = CountVectorizer(
    max_features=None, min_df=0.0007, max_df=0.6000000000000001, ngram_range=(1, 2))
count_vectors = count_vectorizer.fit_transform(
    df_processed_data['PROCESSED_DOC'])

mutual_info_vector = mutual_info_classif(count_vectors, df_processed_data[TARGET_VARIABLE]) 
print(mutual_info_vector)



print(count_vectorizer.get_feature_names())


# ### Model Train and Cross-Validation


count_vectorizer = CountVectorizer(
    max_features=None, min_df=0.0007, max_df=0.6000000000000001, ngram_range=(1, 2))
tfidf_transformer = TfidfTransformer(norm='l1', use_idf=True, sublinear_tf=True)
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


# ### Evaluating the best model


''' Let's evaluate more deeply the best model '''
X_train, X_test, y_train, y_test = train_test_split(
    df_processed_data['PROCESSED_DOC'],
    df_processed_data[TARGET_VARIABLE],
    test_size=0.33, random_state=42)

pipeline = Pipeline([
    ('count_vectorizer', count_vectorizer),
    ('tfidf_transformer', tfidf_transformer),
    ('clf', clf)
])

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


# ### Better visualising model classification


fig = plt.figure(figsize=(20, 20))
axes = plt.axes()

print(plot_confusion_matrix(pipeline, preds, labelsTest1, cmap='hot', ax=axes))


# 
