'''
    This module is created to automatize tests
    functions related to treating data before
    model training and analysis.
    Created by: Thiago Luis
'''


import pytest
import helper_tests
from preprocessing import open_document, document_to_dataframe


@pytest.mark.parametrize('input_and_output', [
    ("45198473.pdf", True),
    ("invalido.pdf", False),
    ("48276987.pdf", True)])
def test_pdf_to_text(input_and_output):
    input_list_string = input_and_output[0]
    expected_output = input_and_output[1]
    extracted_string = open_document(
        input_list_string, helper_tests.TESTS_SAMPLES_RELATIVE_PATH)
    found_page = len(extracted_string) > 0

    assert expected_output == found_page
    ''' Then we test if the text file importation is having any problem '''
    extracted_string = open_document(
        input_list_string, helper_tests.TESTS_SAMPLES_RELATIVE_PATH)

    assert expected_output == found_page
    
@pytest.mark.parametrize('input_and_output', [
    ("45198473.pdf", True),
    ("invalido.pdf", False),
    ("48276987.pdf", True)])
def test_document_to_dataframe(input_and_output):
    input_list_string = input_and_output[0]
    expected_output = input_and_output[1]
    document_pages = open_document(
        input_list_string, helper_tests.TESTS_SAMPLES_RELATIVE_PATH)

    df_document_pages = document_to_dataframe(document_pages, input_and_output)

    assert expected_output == df_document_pages

