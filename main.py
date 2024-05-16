import os
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")

#########################
# STAGE 1: INDEXING DATA #
##########################
# Part 1.1: Loading data


def read_and_extract_text_from_pdf(file_path):
    """
    This function opens the PDF from file, extracts texts therein, and returns it as a string
    :param file_path: (str) Contains a string of the PDF filepath
    :return: (str) Returns a long string of concatenated text extracted from the pdf
    """
    # Create PDF reader object
    reader = PdfReader(file_path)

    # Loop over each page, extract the text, & add it into the text variable
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text


# Call the text scrape & read function
all_pdf_text = read_and_extract_text_from_pdf(file_path="./docs/state_of_the_union.pdf")

