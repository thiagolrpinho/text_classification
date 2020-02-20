import pdftotext
import os
import re
import pandas as pd
from typing import List

DATASETS_RELATIVE_PATH = "./assets/datasets/"


def open_document(
        filename: str, foldername: str = DATASETS_RELATIVE_PATH) -> List[str]:
    '''
    Open a pdf document alterady OCRized, stores a copy of the content as
    a text file(.txt) inside a folder "./txts/" in the same directory as
    the original pdf
    :args: filename -> string, foldername -> string
    :return: document_pages -> List[string]
    '''
    if filename == "txts":
        ''' Stops the function from opening folder as pdf '''
        return []

    filename = re.sub(".pdf", "", filename)
    if not os.path.isfile(foldername + filename + ".pdf"):
        return []

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

    return document_pages


def document_pages_to_dataframe(document_pages: List[str], document_name: str):
    ''' Receives a list of document_pages and convert the pages of
        each document in a row with the  content of the page and
        the document identifier as columns.
        :args: 
        :returns: dataframe containing all pages as rows
    '''
    if not document_pages or not document_name:
        return pd.DataFrame()

    document_pages_dicts = []
    for page in document_pages:
        single_page_dict = {
            "page_content": page,
            "filename": document_name
        }
        document_pages_dicts.append(single_page_dict)

    df_document_pages = pd.DataFrame(document_pages_dicts)

    return df_document_pages
