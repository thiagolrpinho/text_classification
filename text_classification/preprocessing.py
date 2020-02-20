import pdftotext
import os
import re


def open_document(filename: str, foldername: str = "assets/datasets"):
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
                print(
                    "{}"+end_of_page_token.format(page.upper()),
                    file=text_file)
                document_pages.append(page.upper())
    else:
        with open(text_filename, "r") as text_file:
            full_text_content = text_file.read()
            document_pages = full_text_content.split(end_of_page_token)
            # The split function is returning an empty blank page at end of
            # the list so we have to drop it
            document_pages.pop()

    return document_pages
