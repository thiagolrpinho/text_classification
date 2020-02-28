import pdftotext
import os
import re
import pandas as pd
from typing import List, Tuple
from wordcloud import WordCloud
from nltk.tokenize import WordPunctTokenizer
import seaborn as sns
import matplotlib.pyplot as plt

DATASETS_RELATIVE_PATH = "./assets/datasets/"


def open_documents(
        filenames: List[str],
        foldername: str = DATASETS_RELATIVE_PATH) -> List[Tuple[str, List[str]]]:
    '''
    Open multiple pdf documents alterady OCRized, stores a copy of their
    contents as a text file(.txt) inside a folder "./txts/" in the same
    directory as the original pdf
    :args: filenames -> a list containing each pdf filename
            foldername -> a string containing the folder with the documents
                relative path
    :return: document_pages -> List of tuples with filenames and their
            respective pages
    '''
    if not filenames:
        return []
    documents = []
    for filename in filenames:
        if filename == "txts":
            ''' Stops the function from opening folder as pdf '''
            continue

        filename = re.sub(".pdf", "", filename)
        if not os.path.isfile(foldername + filename + ".pdf"):
            continue

        if not os.path.exists(foldername + "txts/"):
            os.makedirs(foldername + "txts/")

        text_filename = foldername + "txts/" + filename + ".txt"
        end_of_page_token = "endofpagemarksymbol"

        if not os.path.exists(text_filename):
            ''' Only reads the pdf if a quicker txt version is not available '''
            with open(foldername + filename + ".pdf", "rb") as pdf_file:
                untreated_document = pdftotext.PDF(pdf_file)
                document_pages = []

            with open(text_filename, "w") as text_file:
                for page in untreated_document:
                    print(page.upper() + end_of_page_token, file=text_file)
                    document_pages.append(page.upper())
        else:
            with open(text_filename, "r") as text_file:
                full_text_content = text_file.read()
                document_pages = full_text_content.split(end_of_page_token)
                # The split function is returning an empty blank page at end of
                # the list so we have to drop it
                document_pages.pop()
        documents.append((filename, document_pages))

    return documents


def documents_pages_to_dataframe(documents: List[Tuple[str, List[str]]]):
    ''' Receives a list of documents with filenames a pages
        and convert the pages of each page in a row with the 
        content of the page and the document identifier as columns.
        If any of the arguments is empty or they're not
        the same size the function will return an empty dataframe
        :args: documents -> Multiple documents with multiple pages
        :returns: dataframe containing all pages as rows
    '''
    if not documents:
        return pd.DataFrame()

    documents_dicts = []
    for document_name, pages in documents:
        for page in pages:
            single_page_dict = {
                "page_content": page,
                "filename": document_name
            }
            documents_dicts.append(single_page_dict)

    df_documents_pages = pd.DataFrame(documents_dicts)

    return df_documents_pages


def generate_wordcloud(df_column, figsize=(20, 8)):
    ''' Receives a single a dataframe column and
        prints on screen a wordcould of the 100
        most common words
        :args: df_column -> Dataframe, figsize -> (int, int)
    '''
    text = df_column.str.cat(sep=" ")
    wordcloud = WordCloud(
        max_words=100,
        width=900,
        height=500,
        max_font_size=350,
        collocations=False,
        normalize_plurals=False).generate(text)
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def generate_freq_dist_plot(df_column, max_words=25):
    ''' Receives a single a dataframe column and
        plots on screen a bar graph with the most
        frequent tokens and also return those tokens
        as a list.
        :args: df_column -> DataFrame, max_words -> int
        :returns: words_that_appear_the_most -> List[int]
    '''
    text = df_column.str.cat(sep=" ")
    words = WordPunctTokenizer().tokenize(text)
    words_that_apper_the_most = pd.Series(
        words).value_counts().nlargest(max_words)

    ax = sns.barplot(
        x=words_that_apper_the_most.index,
        y=words_that_apper_the_most.values)
    ax.figure.set_size_inches(20, 7)

    return words_that_apper_the_most
