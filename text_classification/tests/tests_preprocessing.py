'''
    This module is created to automatize tests
    functions related to treating data before
    model training and analysis.
    Created by: Thiago Luis
'''


import pytest
import helper_tests
from preprocessing import open_document


@pytest.mark.parametrize('input_and_output', [
    ("45198473.pdf", True),
    ("invalido.pdf", False),
    ("48276987.pdf", True)])
def test_pdf_to_text(input_and_output):
    input_list_string = input_and_output[0]
    expected_output = input_and_output[1]
    extracted_string = open_document(
        input_list_string, helper_tests.TESTS_SAMPLES_RELATIVE_PATH)
    assert expected_output == extracted_string
