"""
-*- coding: utf-8 -*-
Created on Fri 21 2020
@author: Thiago Pinho
"""

from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import spacy
from spacy.lang.pt import Portuguese
from spacy.lang.pt.stop_words import STOP_WORDS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


VECTOR_MODEL_NAME = "pt_core_news_sm"
RELATIVE_PATH_TO_FOLDER = "./assets/datasets/ribon/"
DATA_FILENAME = "Feeds_Label"
parser = Portuguese()
nlp = spacy.load(VECTOR_MODEL_NAME)
TARGET_VARIABLE = "LABEL_TRAIN"
TEXT_VARIABLE = "TITLE"

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
data_labels_count = df_ribon_news[TARGET_VARIABLE].value_counts()
data_labels = data_labels_count.index
fig = plt.figure(figsize=(20, 8))
sns.barplot(
    x=data_labels_count.index,
    y=data_labels_count)
plt.show()

"""  Let"s store the data """
excel_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME + "_treated.xlsx"

"""  Convert the dataframe to an xlsx file """
df_ribon_news.to_excel(excel_filename)

"""  We then load the data for stability """
df_ribon_news_treated = pd.read_excel(excel_filename, index_col=0)
print(df_ribon_news_treated.head())

data_labels_with_count = df_ribon_news_treated[TARGET_VARIABLE].value_counts()
data_labels = data_labels_with_count.index
for label in tdqm(data_labels):
    print(label + ": ", data_labels_with_count[label])

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

''' One possible approach is to group up under represented labels and further
    analysis in other pipeline. We analysed the data and found out 
    that some of the labels are under 40 representations which is pretty low.
'''
data_labels_count = df_ribon_news_treated[TARGET_VARIABLE].value_counts()
data_labels = data_labels_count.index
under_represented_labels = [
    scarse_label
    for scarse_label in tdqm(data_labels)
    if data_labels_count[scarse_label] <= 40]
print(under_represented_labels)

''' Now we have found which ones are under represented we'll create a new
    DataFrame changing the under represented to OUTROS '''
GROUP_TARGET_LABEL = 'OUTROS'
df_ribon_news_grouped = df_ribon_news_treated.replace(
    {TARGET_VARIABLE: under_represented_labels}, GROUP_TARGET_LABEL)
print(df_ribon_news_grouped[TARGET_VARIABLE].value_counts())


"""  Let"s see how the labels are distributed """
data_labels_count = df_ribon_news_grouped[TARGET_VARIABLE].value_counts()
data_labels = data_labels_count.index
fig = plt.figure(figsize=(20, 8))
sns.barplot(
    x=data_labels_count.index,
    y=data_labels_count)
plt.show()

""" As we have two text variables CONTENT and TITLE.
    we can use both of then to improve predictions
"""
first_pipeline_text_variable = "CONTENT"
second_pipeline_text_variable = "TITLE"

df_first_ribon_news_data_grouped = df_ribon_news_grouped[
    [first_pipeline_text_variable, TARGET_VARIABLE]]
df_second_ribon_news_data_grouped = df_ribon_news_grouped[
    [second_pipeline_text_variable, TARGET_VARIABLE]]

"""  Let"s store the data """
excel_first_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME +\
    "_first_data_grouped.xlsx"
excel_second_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME +\
    "_second_data_grouped.xlsx"

df_first_ribon_news_data_grouped.to_excel(excel_first_filename)
df_second_ribon_news_data_grouped.to_excel(excel_second_filename)

"""  We then load the data for stability """
df_first_data = pd.read_excel(excel_first_filename, index_col=0)
df_second_data = pd.read_excel(excel_second_filename, index_col=0)
print(df_first_data.head())
print(df_second_data.head())

''' Let's do the pipeline step by step to be more explicit '''
''' Text Parsing
    This part is reponsible for clean the text from symbols and possible
    OCR noise. It's also responsible to find sentences,
    tokenize the text, identifiy(if needed) multi-word terms, find POS
    (Part of Speech), drop unwanted semmantics and stopwords, lemmatize the
    rest.
    - Sentencer
    - Tokenizer
    - Parser
    - Lemmatizer
'''
raw_text_data = df_first_data[first_pipeline_text_variable].to_list()

preprocessed_text_data = [str(raw_text) for raw_text in tqdm(raw_text_data)]
''' Not all variables are being undestood as strings so we have to force it'''

nlp = spacy.load(VECTOR_MODEL_NAME)
''' We an already trained model to process portuguese '''

sentencizer = nlp.create_pipe('sentencizer')
''' Create the pipeline 'sentencizer' component '''

nlp.add_pipe(sentencizer, before='parser')
''' We then add the component to the pipeline '''
print(nlp.pipe_names)

processed_text_data = []
lemmatized_doc = []
for row in tqdm(preprocessed_text_data):
    doc = nlp(row)
    processed_text_data.append(doc)
    lemmatized_doc.append(str([word.lemma_ for word in doc if not word.is_stop]))

df_processed_data = pd.DataFrame()
df_processed_data['preprocessed_text_data'] = preprocessed_text_data
df_processed_data['processed_text_data'] = processed_text_data
df_processed_data['lemmatized_doc'] = lemmatized_doc
print(df_processed_data)


"""  Let"s store the data """
excel_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME +\
    "_parsed_data.xlsx"

df_processed_data.to_excel(excel_filename)

"""  We then load the data for stability """
df_processed_data = pd.read_excel(excel_filename, index_col=0)
print(df_processed_data.head())

''' Best parameter (CV score=0.535):
{'clf__alpha': 1e-05, 'clf__max_iter': 80, 'clf__penalty': 'l2', 'tfidf__norm': 'l1',
'tfidf__use_idf': True, 'vect__max_df': 0.5, 'vect__max_features': None, 'vect__ngram_range': (1, 2)}
'''
''' Text Filter
    This part is responsible to give weights to important tokens and remove
    weight for unwanted ones or those who can be misguiding.
    - Frequency Counter
    - Id-IdF Counter
'''
vect = CountVectorizer(max_features = None, max_df=0.5, vect=ngram_range(1, 2))
tfidf = TfidfTransformer(norm= 'l1', use_idf='True')

''' Text Topics
    This part will be used to generate unsupervisioned topics using the tokens
    from text filter. Those topics will later be used to generate rules for the
    model.
'''


''' Rule Builder

'''

''' Model Train and Evaluation

'''