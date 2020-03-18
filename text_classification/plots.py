import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spacy
from nltk.tokenize import WordPunctTokenizer

nlp = spacy.load("pt_core_news_sm")

def plot_categories_distribution(df, category_variable, id, title="Categories distribution"):
    df_categories = df.groupby(category_variable)[id].count().reset_index().rename(columns={id : 'COUNT'})

    fig = px.bar(df_categories, 
                 x=category_variable, 
                 y='COUNT', 
                 text='COUNT',
                 labels={category_variable: 'Categories',
                         'COUNT': 'Number of Articles'
                 },
                 title=title
                )
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(xaxis_tickangle=45)
    fig.show()
    return df_categories


def plot_term_scatter(df, text_variable, title='Distribution of top-50 terms'):
    '''Plot the distribution of the top-50 terms present in all obs of a text variable 
    and grouping then by part-of-speech'''
    text = df[text_variable].str.cat(sep=" ")
    words = WordPunctTokenizer().tokenize(text)
    '''Selecting top 50 words'''
    top_words = pd.Series(words).value_counts().nlargest(50)
    words_df = top_words.rename_axis('words').reset_index(name='count')
    words_df['pos'] = [nlp(word)[0].pos_ for word in words_df['words']]

    fig = px.scatter(words_df, 
                     x='words', 
                     y='count', 
                     color='pos', 
                     labels={'words':'Terms',
                             'count': 'Presence in Text',
                             'pos': 'Part-of-speech'
                            }, 
                     marginal_y='rug',
                     title=title
                    )
    fig.update_layout(xaxis_tickangle=45, xaxis_categoryorder = 'total descending')
    fig.show()
    return words_df