import os
import re

from dotenv import load_dotenv
from pypdf import PdfReader

import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from typing import List

load_dotenv()


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
extracted_pdf_text = read_and_extract_text_from_pdf(file_path="docs/input_text/state_of_the_union.pdf")


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


# => Part 1.3: Create Embeddings
# Embeddings: Vector representations of the text to enable math operations
# Text embedding tool: An embedding class (extends chromadb embed func.) that makes API calls to Gemini embedding model
# The embedding class: Also used by chromadb vector DB client for indexing & querying.


# Reference: https://cookbook.chromadb.dev/embeddings/bring-your-own-embeddings/
class GeminiEmbeddingFunction(EmbeddingFunction):
    # TODO: Add return type to __call__ method
    def __call__(self, input: Documents):
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API key not found. Please provide one!")
        genai.configure(api_key=gemini_api_key)
        model = 'models/embedding-001'
        title = 'Custom prompt'
        text_embeddings = genai.embed_content(model=model,
                                              content=input,
                                              task_type='retrieval_document',
                                              title=title)['embedding']
        return text_embeddings


# => Part 1.4: Create & Store Embeddings
# Here, a chromadb client is defined & stores text_embeddings persistently in a defined filepath
def create_chroma_vector_db(documents: List, path: str, collection_name: str):
    """
    Creates Chromadb loaded with the document embeddings collection at specified path, named with the provided name arg
    :param documents: Documents iterable with text chunks to be converted to embeddings & added to the Chromadb
    :param path: Filepath where the Chromadb shall be stored
    :param collection_name: Name of the created vector embeddings collection added to the Chromadb
    :return: A tuple containing the created Chroma collection & its name
    """
    chromadb_client = chromadb.PersistentClient(path=path)

    # Check if collection_name already exists in the list of collections
    is_present = any(collection.name == collection_name for collection in chromadb_client.list_collections())

    if not is_present:
        db = chromadb_client.create_collection(name=collection_name, embedding_function=GeminiEmbeddingFunction())

        for idx, doc in enumerate(documents):
            db.add(documents=doc, ids=str(idx))
        return db, collection_name
    else:
        return None, None  # Return none if the collection already exists


chroma_collection, name = create_chroma_vector_db(documents=paragraph_chunks,
                                                  path='./docs/chromadb_collections',
                                                  collection_name='state_of_union')


# => Part 1.5: Chromadb collection loader
# This function uses a chromadb client to get/load an already created ChromaDB collection from DB using its name
def read_chroma_vector_db_collection(path, collection_name):
    chromadb_client = chromadb.PersistentClient(path=path)
    found_db = chromadb_client.get_collection(name=collection_name, embedding_function=GeminiEmbeddingFunction())

    return found_db

