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

vectors.shape

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