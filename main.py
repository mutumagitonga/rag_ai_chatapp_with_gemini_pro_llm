import os
import re

from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")

#########################
# STAGE 1: INDEXING DATA #
##########################
# => Part 1.1: Loading data


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
extracted_pdf_text = read_and_extract_text_from_pdf(file_path="./docs/state_of_the_union.pdf")


# => Part 1.2: Splitting data
# The data is split into many chunks to ensure shorter context windows (recipe for better accuracy)
# However, the reason above doesn't help much for Gemini Pro since it currently has a context window of 1M tokens
# Thus, the splitting speeds up the 1st stage of the RAG process when a user prompt is used to search for ...
# ... semantically (relevant) content from the user text corpus before creating an augmented prompt for the LLM API


def split_text_corpus_every_paragraph(text_corpus):
    """
    This function splits the previously extracted text string at every paragraph using regex marker "\n \n"
    :param text_corpus: (type: str) The text string to split into individual paragraphs
    :return: (List[str]) Returns a list of strings representing split paragraphs
    """
    split_paragraphs = re.split('\n \n', text_corpus)
    return [paragraph for paragraph in split_paragraphs if paragraph != ""]  # List comprehension to create list


# Call the paragraph-splitting function to return a list of paragraph strings
paragraph_chunks = split_text_corpus_every_paragraph(text_corpus=extracted_pdf_text)
