#!/usr/bin/env python
# coding: utf-8

# # Text classification, an SAS EM inspired approach

# ## Summary
# 
# 1. Load Data
# 2. Exploratory analysis
# 3. Data preparation  
#     3.1. Renaming the columns  
#     3.2. Dropping unwanted variable  
#     3.3. Selecting input variable  
#     3.4. Dropping missing values  
#     3.5. Balancing target categories for model input  
# 4. Text Filter  
#     4.1. Removing ponctuation and stopwords  
#     4.2. Lemmatizing and stemming  
#     4.3. Entity recognition and filtering  
#     4.4. Part-of-Speech analysis and filtering  
#     4.5. Analysing filtering results  
# 5. Text Parser
#     5.1 Dropping missing entities and POS
#     5.2 Counting and Vectorizing
#     5.3 Grisearching Parameters #TODO não sei se esse tópico precisa ser um tópico a parte, eu tentei ligar os textos usando somente texto ao invés de títulos.
# 6. Topic Modelling
#     6.1 Generating Topics
#     6.2 Grisearching Parameters
#     6.3 Storing topics scores as variables
# 7. Model Train and Cross-Validation
#     7.1 Comparing models
#     7.2 Evaluating winner model
#     7.3 Confusion Matrix

# In[48]:


"""
-*- coding: utf-8 -*- Created on Fri 21 2020
@author: Thiago Pinho, Gabriel Vasconcelos
@colaborators: Thiago Russo, Emmanuel Perotto
"""
#todo limpar algumas libs ai
from sklearn.model_selection import cross_val_score, train_test_split,\
    GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report,\
    plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pyLDAvis
from pyLDAvis import sklearn as sklearn_lda
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
from nltk.corpus import stopwords
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plots import plot_term_scatter, plot_categories_distribution,\
    plot_wordcloud
warnings.simplefilter("ignore", DeprecationWarning)


# For better code management, the constants used in this notebook will be listed bellow.

# In[4]:


RELATIVE_PATH_TO_FOLDER = "./assets/datasets/"
DATA_FILENAME = "feeds_label"
VECTOR_MODEL_NAME = "pt_core_news_sm"
NLP_SPACY = spacy.load(VECTOR_MODEL_NAME)
TARGET_VARIABLE = "LABEL_TRAIN"
POSSIBLE_TEXT_VARIABLES = ["CONTENT", "TITLE"]


# ## 1. Load data
# 
# We'll have a look at the raw data and analyse it's structure.

# In[7]:


relative_path_file = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME + ".csv"
ribon_news_df = pd.read_csv(relative_path_file)

ribon_news_df.head()


# ## 2. Exploratory analysis
# 
# First we'll check datatypes and number of rows and columns

# In[8]:


print("Data info:")
ribon_news_df.info()


# We can see that we have two text variables that we can use as input and some missing values in the 'content' variable.
# 
# Let's see how the target variable values are distributed

# In[9]:


raw_categories_df = plot_categories_distribution(ribon_news_df,
                                                 'Label_Train', 
                                                 'id', 
                                                 'Initial distribution of Categories in the Ribon News dataset')


# We can notice two things:
# 
# 1. The column labels are not all uppercase or lowercase. 
# 2. The categories avaiable to classify are not all in the same case either which could lead to later confunsion on the real number of categories the model should classify.

# ## 3. Data preparation
# 
# ### 3.1. Renaming the columns
# 
# Let's start uppercasing all column names and target variable values

# In[10]:


ribon_news_df.columns = map(lambda x: str(x).upper(), ribon_news_df.columns)
ribon_news_df[TARGET_VARIABLE] = ribon_news_df[TARGET_VARIABLE].str.upper()

ribon_news_df.head()


# Now let's the real distribution of categories

# In[11]:


df_categories2 = plot_categories_distribution(ribon_news_df, 
                                              TARGET_VARIABLE, 
                                              'ID',
                                              'Distribution of Categories after preparing the text'
                                             )


# We can see that categories like 'CRIANCAS', 'IDOSOS' and 'FAMILIA' maybe have too few observations to train the model and the 'ECOLOGIA' category has too many observations, which can bias the model. 
# 
# Before processing the data, let's store our partial progress for more stability and less rework.

# ### 3.2. Dropping unwanted variables
# 
# Not all columns available in data will be useful for the label classification.

# In[12]:


preprocessed_data_df = ribon_news_df.drop(['PICK_COUNT', 'ID'], 1)
preprocessed_data_df.info()


# ### 3.3. Selecting input variable
# 
# In the previous results, we noticed that are two text variables besides the target: 'CONTENT' and 'TITLE'. As 'CONTENT' is empty in two cases but 'TITLE' is not empty in any case.
# 
# Lets analyse their size so we can decide which one is better as our model input.

# In[13]:


print('Column TITLE mean length is ', int(preprocessed_data_df['TITLE'].str.len().mean()),
      ' and standard deviation was ', int(preprocessed_data_df['TITLE'].str.len().std())) 
print('Column CONTENT mean length is ', int(preprocessed_data_df['CONTENT'].str.len().mean()),
      ' and standard deviation was ', int(preprocessed_data_df['CONTENT'].str.len().std())) 


# The variable CONTENT appears to have more data and could bring better results. But as two rows have this column empty we would have to drop those. One way around it is to oversample the data by using both as text variables.
# 

# In[14]:


data_df = pd.concat([
    preprocessed_data_df[['TITLE', TARGET_VARIABLE]].rename(columns={'TITLE':"TEXT_VARIABLE"}), 
    preprocessed_data_df[['CONTENT', TARGET_VARIABLE]].rename(columns={'CONTENT':"TEXT_VARIABLE"})
])
data_df.info()


# ### 3.4. Dropping missing values
# 
# As there are some samples that are empty, they'll not be useful to train or to validate the model. 

# In[15]:


data_df = data_df.dropna()
data_df.info()


# Lets see how our dataset looks like now

# In[16]:


data_df['ID'] = data_df.index
df_categories3 = plot_categories_distribution(data_df, 
                                              TARGET_VARIABLE, 
                                              'ID',
                                              'Distribution of categories after oversampling'
                                             )


# ### 3.5 Balancing target categories for model input
# 
# One important step is to analyse how the target categories are distributed, so we can better partition our data.

# In[17]:


categories_df = data_df[TARGET_VARIABLE].value_counts().reset_index()
categories_mean = categories_df[TARGET_VARIABLE].mean()
categories_std = categories_df[TARGET_VARIABLE].std()
print("Mean number of samples of each category is: ", categories_mean)
print("Standard deviation number of samples of each category is: ", categories_std)

over_categories = [obs[0] for obs in categories_df.values if obs[1] > categories_mean+categories_std]
under_categories = [obs[0] for obs in categories_df.values if obs[1] < categories_mean-categories_std]
print("\nOverrepresented categories:", over_categories)
print("Underrepresented categories:", under_categories)
print("PS: The outliers were calculated using the 1 standart deviation")


# The categories "ECOLOGIA" and "SOLIDARIEDADE" are overrepresented which may cause the model to overly recognise those labels patterns and make then too sensitive for those. 
# 
# On the other hand we have the categories "FAMILIA", "CRIANCAS" and "IDOSOS" that are under represented, which can make the model too specific for those and hardly classify as it.
# 
# An easy sollution is to group the least common labels into a single one. When our pipeline is finely tunned we can use the grouped labels as input for another pipeline trained only to discern among those. For the overrepresented labels, a way out is to randomly select just a sample of them.

# In[18]:


data_df.loc[data_df[TARGET_VARIABLE].isin(under_categories), TARGET_VARIABLE] = "OTHERS"

#for category in over_categories:
top_limit = categories_mean + categories_std
grouped_data_df = data_df[~data_df[TARGET_VARIABLE].isin(over_categories)]
for over in over_categories:
    rejected_obs = int(data_df[TARGET_VARIABLE].value_counts()[over] - top_limit)
    over_df = data_df[data_df[TARGET_VARIABLE] == over].sample(int(top_limit))
    grouped_data_df = grouped_data_df.append(over_df)


# Let's see the result

# In[19]:


grouped_data_df['ID'] = grouped_data_df.index
df_categories4 = plot_categories_distribution(grouped_data_df, 
                                              'LABEL_TRAIN', 
                                              'ID',
                                              'Distribuition of categories after grouping and filtering'
                                             )


# ### 3.7. Visual analysis of pre-processing

# In[20]:


fig = go.Figure(data=[
    go.Bar(name='First dataset (uppercased)', 
           x=df_categories2['LABEL_TRAIN'], 
           y=df_categories2['COUNT']
    ),
    go.Bar(name='Second dataset (oversampled)', 
           x=df_categories3['LABEL_TRAIN'], 
           y=df_categories3['COUNT']
    ),
    go.Bar(name='Third dataset (filtered and grouped)', 
           x=df_categories4['LABEL_TRAIN'], 
           y=df_categories4['COUNT']
    )
])

fig.update_layout(
    barmode='group',
    title="Evolution of the dataset during the pre-processing",
    xaxis_title="Categories",
    yaxis_title="Number of Articles + Titles"
)

fig.show()


# To better manage the step-by-step of the model evolution, let's store our partial progress

# In[21]:


excel_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME + "_treated_grouped.xlsx"
grouped_data_df.to_excel(excel_filename)


# ## 4. Text Filter
# 
# Before we train the model, it's necessary to create a bag of words by tokenizing each word, finding their lemmas and discarting some words that could mislead the model.
# 
# Let's take a first look at the raw text variable.

# In[22]:


"""  We then load the data for stability """
data_df = pd.read_excel(excel_filename, index_col=0)
data_df.info()


# In[23]:


text_variable = 'TEXT_VARIABLE'
df_terms1 = plot_term_scatter(data_df, text_variable, 'Top-100 terms in raw data')


# ### 4.1. Removing ponctuation and stopwords
# 
# As we can see, we have a lot of tokens from text variable being ponctuations or words that don't have by themselves much meaning. 
# 
# We're going to load a built-in stopwords list to remove these unnecessary tokens.

# In[24]:


stopwords_set = set(STOP_WORDS).union(set(stopwords.words('portuguese')))                               .union(set(['anos', 'ano', 'dia', 'dias']))
    
print("This is the stopword list: ", sorted(list(stopwords_set)))


# To improve performance, we're going to use the loop in the next step to effectively remove the stopwords from text

# ### 4.2. Lemmatizing and stemming
# 
# #todo explicar o role

# In[25]:


''' Not all variables are being undestood as strings so we have to force it'''
preprocessed_text_data = data_df[text_variable].to_list()
''' Create the pipeline 'sentencizer' component '''
sentencizer = NLP_SPACY.create_pipe('sentencizer')
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
    preprocessed_doc = [token for token in doc if token.is_alpha and not token.norm_ in stopwords_set]
    tokenized_data.append(preprocessed_doc)
    raw_doc.append(" ".join([word.text for word in preprocessed_doc]))
    lemmatized_doc.append(" ".join([word.lemma_ for word in preprocessed_doc]))
    normalized_doc.append(" ".join([word.norm_ for word in preprocessed_doc]))

data_df['RAW_DOC'] = raw_doc
data_df['NORMALIZED_DOC'] = normalized_doc
data_df['LEMMATIZED_DOC'] = lemmatized_doc

data_df.head()


# Visual plotting of the top-100 terms in raw text, normalized text and lemmatized text, respectively.

# In[26]:


'''Visual plotting of terms pre-processing'''
df_terms2 = plot_term_scatter(data_df, 'RAW_DOC', 'Top-100 terms of tokenized data without lemmatizaton')
df_terms3 = plot_term_scatter(data_df, 'NORMALIZED_DOC', 'Top-100 terms of normalized data with minor lemmatization')
df_terms4 = plot_term_scatter(data_df, 'LEMMATIZED_DOC', 'Top-100 terms of with full lemmatization')


# ### 4.3. Entity recognition and filtering
# 
# Some parts of speech may mislead the model associating classes to certain entities that are not really related to the categories.
# The NER model (spacy portuguese) we are using uses the following labels:
# 
# | TYPE | DESCRIPTION |
# |------|-------------------------------------------------------------------------------------------------------------------------------------------|
# | PER | Named person or family. |
# | LOC | Name of politically or geographically defined location (cities, provinces, countries, international regions, bodies of water, mountains). |
# | ORG | Named corporate, governmental, or other organizational entity. |
# | MISC | Miscellaneous entities, e.g. events, nationalities, products or works of art. |
# 
# Let's take a look at all entities present in the text

# In[27]:


entities_obs = []
for doc in tokenized_data:
    for token in doc:
        if token.ent_type_:
            entities_obs.append((token.text, token.ent_type_))

entities_df = pd.DataFrame(entities_obs, columns=['entity_text', 'entity_type'])
entities_df['id'] = entities_df.index
grouped_entities_df = (entities_df.groupby(['entity_text', 'entity_type']).count()
                                                                         .reset_index()
                                                                         .rename(columns={'id': 'count'})
                                                                         .sort_values('count', ascending=False)
                                                                         .head(100))
                
fig = px.treemap(grouped_entities_df, 
                 path=['entity_type', 'entity_text'], 
                 values='count', 
                 title='Top-50 entities present in text grouped by type')
fig.show()
entity_type_df = grouped_entities_df['entity_type'].value_counts().reset_index()

fig = px.pie(entity_type_df, values='entity_type', names='index', title='Entity type frequency')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# #todo explicar porque removemos pessoas e organizações

# In[28]:


processed_tokenized_data = []
processed_doc_text = []
entities_obs = []
entity_unwanted_types = set(['PER', 'ORG'])

for doc in tokenized_data:
    entities_text = ""
    processed_doc = []
    for token in doc:
        if not token.ent_type_:
            processed_doc.append(token)
        elif not token.ent_type_ in entity_unwanted_types:
            processed_doc.append(token)
            entities_obs.append((token.text, token.ent_type_))
        
    processed_tokenized_data.append(processed_doc)
    processed_doc_text.append(" ".join([word.norm_ for word in processed_doc ]))

''' Processing text on entity level'''
data_df['PROCESSED_DOC'] = processed_doc_text
data_df.head()


# Visual analysis of the remaining entities

# In[29]:


entities_df = pd.DataFrame(entities_obs, columns=['entity_text', 'entity_type'])
entities_df['id'] = entities_df.index
grouped_entities_df = (entities_df.groupby(['entity_text', 'entity_type']).count()
                                                                         .reset_index()
                                                                         .rename(columns={'id': 'count'})
                                                                         .sort_values('count', ascending=False)
                                                                         .head(100))
fig = px.treemap(grouped_entities_df, 
                 path=['entity_type', 'entity_text'], 
                 values='count', 
                 title='Top-50 entities present in text grouped by type after filtering')
fig.show()

entity_type_df = grouped_entities_df['entity_type'].value_counts().reset_index()
fig = px.pie(entity_type_df, values='entity_type', names='index', title='Entity type frequency after filtering')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# ### 4.4 Part-of-Speech analysis and filtering
# Let's take a look in the parts of speech presents in the dataset

# In[30]:


token_obs = []
for doc in processed_tokenized_data:
    for token in doc:
        token_obs.append((token.norm_, token.pos_))

token_df = pd.DataFrame(token_obs)
token_df.columns = ['token', 'pos']
token_df['id'] = token_df.index

''' Plotting token grouped by POS treemap '''
grouped_token_df = (token_df.groupby(['token', 'pos']).count()
                                                     .reset_index()
                                                     .rename(columns={'id': 'count'})
                                                     .sort_values('count', ascending=False)
                                                     .head(100))

fig = px.treemap(grouped_token_df, 
                 path=['pos', 'token'], 
                 values='count', 
                 title='Top-100 tokens present in text grouped by Part-of-Speech')
fig.show()

''' Plotting POS bar chart '''
pos_df = token_df.groupby('pos').count().reset_index().rename(columns={'token':'count'})

fig = px.bar(pos_df, x='pos', y='count', text='count', title="Part-of-Speech type frequency")
fig.update_layout(xaxis_categoryorder = 'total descending')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.show()


# #todo explicar grafico acima e a escolha de POS, fazer uma tebela do significado das siglas (parecido com as entidades ou um link pra doc do spacy)
# 
# Here is a wordcloud showing a sample of not relevant words to the text classification that are present in our text variable.

# In[31]:


token_df['id'] = token_df.index
grouped_token_df = token_df.groupby(['token', 'pos']).count().reset_index().rename(columns={'id': 'count'})

plot_wordcloud(grouped_token_df[~grouped_token_df['pos'].isin(set(["PROPN", "NOUN", "ADV", "ADJ", "VERB"]))]['token'])


# Now we're going to remove them, only allowing proper nouns, nouns, adjectives, adverbs and verb to present in our text variable.

# In[32]:


allowed_pos_set = set(["PROPN", "NOUN", "ADV", "ADJ", "VERB"])

processed_doc = []
filtered_token_obs = []
for doc in processed_tokenized_data:
    doc_tokens = [word for word in doc if str(word.pos_) in allowed_pos_set]
    filtered_token_obs.append(doc_tokens)
    processed_doc.append(" ".join(token.norm_ for token in doc_tokens))

data_df['PROCESSED_DOC'] = processed_doc
data_df['TOKENS'] = filtered_token_obs
data_df.head()


# Removing extra spaces originated from the removal of tokens

# In[33]:


space_pattern = r'\s\s+'
data_df['PROCESSED_DOC'] = data_df['PROCESSED_DOC'].str.replace(space_pattern, " ").str.strip()


# Now let's have a look in the tokens after filtering the unwanted POS

# In[34]:


token_obs = []
for doc in filtered_token_obs:
    for token in doc:
        token_obs.append((token.norm_, token.pos_))

token_df = pd.DataFrame(token_obs)
token_df.columns = ['token', 'pos']
token_df['id'] = token_df.index

''' Plotting token grouped by POS treemap '''
grouped_token_df = (token_df.groupby(['token', 'pos']).count()
                                                     .reset_index()
                                                     .rename(columns={'id': 'count'})
                                                     .sort_values('count', ascending=False)
                                                     .head(100))

fig = px.treemap(grouped_token_df, 
                 path=['pos', 'token'], 
                 values='count', 
                 title='Top-100 tokens present in text grouped by Part-of-Speech after filtering')
fig.show()

''' Plotting POS bar chart '''
pos_df = token_df.groupby('pos').count().reset_index().rename(columns={'token':'count'})

fig = px.bar(pos_df, x='pos', y='count', text='count', title="Part-of-Speech type frequency after filtering")
fig.update_layout(xaxis_categoryorder = 'total descending')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.show()


# ### 4.5. Analysing filtering results
# 
# Now that we filtered every unwanted stopwords, entity and POS of the orginal text, let's see the most common words for each label and try to check if they make sense.

# In[35]:


category_tokens_obs_df = data_df.groupby('LABEL_TRAIN')['TOKENS'].sum().reset_index()

category_tokens_df = category_tokens_obs_df.explode('TOKENS')
category_tokens_df.reset_index(drop=True, inplace=True)
category_tokens_df['ID'] = category_tokens_df.index

category_tokens_df['TOKENS'] = category_tokens_df['TOKENS'].map(lambda token: token.norm_)

grouped_category_tokens_df = (category_tokens_df.groupby(['LABEL_TRAIN', 'TOKENS']).count()
                                                                                   .reset_index()
                                                                                   .rename(columns={'ID': 'COUNT'})
                                                                                   .sort_values('COUNT', ascending=False))

sampled_category_tokens_df = pd.DataFrame()
for category in grouped_category_tokens_df['LABEL_TRAIN'].unique():
    sampled_category_tokens_df = sampled_category_tokens_df.append(
        grouped_category_tokens_df[grouped_category_tokens_df['LABEL_TRAIN']  == category].nlargest(10, 'COUNT')
    )

fig = px.treemap(sampled_category_tokens_df, 
                 path=['LABEL_TRAIN', 'TOKENS'], 
                 values='COUNT', 
                 title='Top-10 tokens present in each category')
fig.show()


# Now that we finished our filtering, let's store our partial progress

# In[36]:


excel_filename = RELATIVE_PATH_TO_FOLDER + DATA_FILENAME + "_processed_data.xlsx"
data_df.to_excel(excel_filename)


#  ## 5. Text Parser
#  
#  Now we have clear text variables we can measure how much they influence the outcome prediction and how many of them exist in each sample.

# In[37]:


processed_data_df = pd.read_excel(excel_filename, index_col=0)
print(data_df.info())


# ### 5.1. Dropping missing Entities and POS
# As there are some samples without content, they'll not be useful to train or to validate the model. 
# Hapilly they're not many so let's drop them.

# In[38]:


processed_data_df = processed_data_df.drop(columns=['TOKENS']).dropna()
print(processed_data_df.info())


# ### 5.2. Counting and Vectorizing
# 
# As our model works only with numerical data we need to convert the string tokens to numeric equivalents which are called features. This part is responsible to give weights to important tokens and remove weight for unwanted ones or those who can be misguiding.
# Let's first instantiate the models used for this process. 
# CountVectorizer generates weights relative to how many times a word or a combination or words(ngrams) appear no matter how big is the document.
# While TfidfTransformer makes it proportional to the size of the document. The parm "use_idf" highlights the less
# frequents ones because they can be more informative than other words that appear a lot.

# In[54]:


''' Best parameter using GridSearch (CV score=0.535): 
{'clf__alpha': 1e-05, 'clf__max_iter': 80, 'clf__penalty': 'l2', 'tfidf__norm': 'l1',
'tfidf__smooth_idf': False, 'tfidf__sublinear_tf': True, 'tfidf__use_idf': True,
'vect__max_df': 0.6000000000000001, 'vect__max_features': None, 'vect__min_df': 0.0007,
'vect__ngram_range': (1, 2)}
Those were obtained on the next code block.
'''
count_vectorizer = CountVectorizer(
    max_features=None, min_df=0.0007, max_df=0.6, ngram_range=(1, 2))
tfidf_transformer = TfidfTransformer(norm='l1', use_idf=True, sublinear_tf=True)

''' Let's transform the lemmatized documents into count vectors '''
count_vectors = count_vectorizer.fit_transform(
    processed_data_df['PROCESSED_DOC'])

''' Then use those count vectors to generate frequency vectors '''
frequency_vectors = tfidf_transformer.fit_transform(count_vectors)

print(processed_data_df['PROCESSED_DOC'].to_list()[0])
print(count_vectors[0]) #TODO talvez tenha uma forma mais informativa de mostrar essa matriz esparsa
print(frequency_vectors[0]) #TODO Dessa também, o objetivo é mostrar que os tokens 


# To find which params where most adequate to vectorize our data we can use a gridsearch algorithm. It brute force tests all the choosen pipeline with the search params so if we have 2 params for one model and 3 for other, it'll test 6 different pipelines and choose the one with best accuracy results using cross-validation. This consumes a lot of time so we use a boolean to choose when to control when to execute this part.

# In[40]:


is_gridsearching = False
if is_gridsearching:
    search_count_vectorizer = CountVectorizer()
    search_tfidf_transformer = TfidfTransformer()
    clf = SGDClassifier(alpha=1e-05, max_iter=80, penalty='l2')
    
    ''' Those are all the params values that will be tested.'''
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
    gs.fit(data_df['PROCESSED_DOC'].values, data_df[TARGET_VARIABLE])
    results = gs.cv_results_
    print(gs.best_params_)


# Now we have choosen the best parameters and already vectorized our data we can take a look on what combinations of tokens did the vectorizer generated as features. This is a good way to be sure the tokenization worked as planned.

# In[41]:


print(count_vectorizer.get_feature_names())


# #TODO Eu queria mostrar quais combinações de tokens os vetorizadores usaram como features.

# ## 6. Topic Modelling
# One way to organize those feature vectors is to search for unsupervisionised patterns inside data to form topics and then use those topics to classify.

# ### 6.1. Generating topics

# In[42]:


num_topics = 30
number_words = 10
''' Creating and fit the LDA model using the count_vectors generated before '''
lda = LDA(n_components=num_topics, max_iter = 20, n_jobs=-1)
topics_vectors = lda.fit_transform(count_vectors)
''' Printing the topics found by the LDA model '''
print("Topics found via LDA:")

words = count_vectorizer.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("\nTopic #%d:" % topic_idx)
    print(" ".join([words[i]
                    for i in topic.argsort()[:-number_words - 1:-1]]))


# Let's see how the topics found are related to each other

# In[49]:


LDAvis_prepared = sklearn_lda.prepare(lda, count_vectors, count_vectorizer)

pyLDAvis.display(LDAvis_prepared)


# ### 6.2. GridSearching best params
# 
# Same as before, we can search for the best params to generate the topics using gridsearch. 

# In[50]:


is_gridsearching = False
if is_gridsearching:
    search_count_vectorizer = CountVectorizer(
        max_features=None, min_df=0.0007, max_df=0.6, ngram_range=(1, 2))
    search_tfidf_transformer = TfidfTransformer(norm='l1', use_idf=True, sublinear_tf=True)
    lda = LDA(n_jobs=-1)
    gnb = GaussianNB()

    search_params = {
        'lda__max_iter': [20],
        'lda__n_components': [60]
    } 

    search_pipeline = Pipeline([
        ('vect', search_count_vectorizer),
        ('tfidf', search_tfidf_transformer),
        ('lda', lda),
        ('gnb', gnb)
    ])

    gs = GridSearchCV(search_pipeline,
                      param_grid=search_params, cv=5)
    gs.fit(processed_data_df['PROCESSED_DOC'].values, processed_data_df[TARGET_VARIABLE])
    print(gs.best_params_)


# ### 6.3. Storing topics scores as variables

# In[55]:


''' We'll need the same number of lists as the number of topics  '''
topics_scores = [[] for i in range(num_topics)]

''' We then extract each row score to different columns '''
for doc_topics in topics_vectors:
    for i in range(0, num_topics):
        if doc_topics[i]:
            topics_scores[i].append(doc_topics[i])
        else: 
            topics_scores[i].append(0)


''' And store then in the data as variables for their respectives rows'''
topics_skl_columns = []
for i in range(0, num_topics):
    column_name = 'TOPIC_SKL_' + str(i)
    topics_skl_columns.append(column_name)
    processed_data_df[column_name] = topics_scores[i]

print(processed_data_df.info())


# ## 7. Model Train and Cross-Validation
# Let's try the analysis with topic modeling or using the vectorized features frequencies. 
# ### 7.1 Comparing models
# #### Topic Modeling

# In[56]:


classifier = RandomForestClassifier(max_depth=4, random_state=0)

pipeline_simple = Pipeline([
    ('classifier', classifier)
])

''' Let's use cross validation to better evaluate models ''' 
scores = cross_val_score(
    pipeline_simple,
    processed_data_df[topics_skl_columns],
    processed_data_df[TARGET_VARIABLE], cv=5)
print("Mean accuracy for explicit pipeline: ", scores.mean()) #TODO uma forma de visualizar esse resultado melhor que um print


# #### Features Vectors Inversed Frequencies

# In[58]:


count_vectorizer = CountVectorizer(
    max_features=None, min_df=0.0007, max_df=0.6000000000000001, ngram_range=(1, 2))
tfidf_transformer = TfidfTransformer(norm='l1', use_idf=True, sublinear_tf=True)
clf = SGDClassifier(alpha=1e-05, max_iter=80, penalty='l2')

''' Encapsuling components in pipeline '''
pipeline = Pipeline([
    ('count_vectorizer', count_vectorizer),
    ('tfidf_transformer', tfidf_transformer),
    ('clf', clf)
])

scores = cross_val_score(
    pipeline,
    processed_data_df['PROCESSED_DOC'],
    processed_data_df[TARGET_VARIABLE], cv=10)
print("Mean accuracy for pipeline: ", scores.mean())


# ### 7.2 Evaluating the winner model

# In[61]:


''' Let's evaluate more deeply the best model '''
X_train, X_test, y_train, y_test = train_test_split(
     processed_data_df['PROCESSED_DOC'].to_list(),
    processed_data_df[TARGET_VARIABLE].to_list(),
    test_size=0.33, random_state=42)

''' First we need to instantiate some components again to avoid overfit'''
count_vectorizer = CountVectorizer(
    max_features=None, min_df=0.0007, max_df=0.6000000000000001, ngram_range=(1, 2))
tfidf_transformer = TfidfTransformer(norm='l1', use_idf=True, sublinear_tf=True)
clf = SGDClassifier(alpha=1e-05, max_iter=80, penalty='l2')

''' Encapsuling components in pipeline '''
pipeline = Pipeline([
    ('count_vectorizer', count_vectorizer),
    ('tfidf_transformer', tfidf_transformer),
    ('clf', clf)
])

train1 = X_train
labelsTrain1 = y_train
test1 = X_test
labelsTest1 = y_test
"""  train """
pipeline.fit(train1, labelsTrain1)
"""  test """
preds = pipeline.predict(test1)
print("accuracy:", accuracy_score(labelsTest1, preds))
print(
    classification_report(
        labelsTest1,
        preds,
        target_names=processed_data_df[TARGET_VARIABLE].unique()))


# ### 7.3 Confusion Matrix

# In[63]:


fig = plt.figure(figsize=(20, 20))
axes = plt.axes()

plot_confusion_matrix(pipeline, test1, labelsTest1, cmap='hot', ax=axes)


# 
