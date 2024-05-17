import os
import re

from dotenv import load_dotenv
from pypdf import PdfReader
import pandas as pd

import google.generativeai as genai
from google.generativeai import GenerationConfig
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from typing import List

load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)


##########################
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
        page_text = page.extract_text()
        if page_text:
            text += page_text
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
paragraph_chunks = split_text_corpus_every_paragraph(text_corpus=extracted_pdf_text)  # 197 chunks

# for m in genai.list_models():
#     if 'embedContent' in m.supported_generation_methods:
#         print(m.name)  # models/embedding-001 models/text-embedding-004


# => Part 1.3: Create Embeddings
# Embeddings: Vector representations of the text to enable math operations
# Text embedding tool: An embedding class (extends chromadb embed func.) that makes API calls to Gemini embedding model
# The embedding class: Also used by chromadb vector DB client for indexing & querying.


# Reference: https://cookbook.chromadb.dev/embeddings/bring-your-own-embeddings/
class GeminiEmbeddingFunction(EmbeddingFunction):
    # TODO: Add return type to __call__ method
    def __call__(self, input: Documents) -> Embeddings:
        if not gemini_api_key:
            raise ValueError("Gemini API key not found/invalid. Please provide one!")
        model = 'models/text-embedding-004'
        title = 'Custom prompt'
        text_embeddings = genai.embed_content(model=model,
                                              content=input,
                                              task_type='retrieval_document',
                                              title=title)['embedding']
        return text_embeddings


# => Part 1.5: Chromadb collection loader
# DEFINE LOADER BEFORE CREATOR function to LOAD any EXISTING CHROMADB COLLECTIONS in create function
# This function uses a chromadb client to get/load an already created ChromaDB collection from DB using its name
def read_chroma_vector_db_collection(path, collection_name):
    try:
        chromadb_client = chromadb.PersistentClient(path=path)
        found_db = chromadb_client.get_collection(name=collection_name, embedding_function=GeminiEmbeddingFunction())
        return found_db
    except ValueError:
        print(f"Collection {collection_name} does not exist.")


# => Part 1.4: Create & Store Embeddings
# Here, a chromadb client is defined & stores text_embeddings persistently in a defined filepath
def create_chroma_vector_db(documents, path, collection_name):
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
        return db
    else:
        found_db = read_chroma_vector_db_collection(path='./docs/chromadb_collections',
                                                    collection_name=collection_name)
        return found_db  # Return the existing collection


union_db = create_chroma_vector_db(documents=paragraph_chunks,
                                   path='./docs/chromadb_collections',
                                   collection_name='state_of_union')
# print(union_db.count())
# print(pd.DataFrame(union_db.peek(8)))


####################################
# STAGE 2: RETRIEVAL & GENERATION ##
####################################
# => Part 2.1: Retrieving relevant data chunks
def get_relevant_text_passage(query, db, n_results):
    relevant_passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
    return relevant_passage


# question = "What are some sanctions on Russia"
# relevant_text_chunk = get_relevant_text_passage(query=question, db=union_db, n_results=6)
# print(relevant_text_chunk)

####
# => Part 2.2: Generating answer using Augmented prompt  ###
####
# (User query + Relevant text chunks) = augmented prompt -> Gemini Pro LLM = Context aware final response

# => Part 2.2.1: Define the prompt template
def rag_augmented_prompt_template(query, relevant_text):
    cleaned_text = relevant_text.replace("'", "").replace('"', '').replace("\n", " ")
    prompt = (
        f"""
        Your are a helpful assistant that answers user queries based on the the contextual knowledge provided in the 
        reference text appended below. Please provide plausible responses in grammatically correct sentences
        relying on the background information. Also, let your responses be simple since 
        your audience does not have a professional foundation on the matters in question. 
        If a user question is not at all related to the reference text given below, ignore it and politely inform the 
        user that you cannot find the answer in the given context.  
        QUESTION: '{query}'
        REFERENCE: '{relevant_text}'
        
        ANSWER:
        """).format(query=query, relevant_text=cleaned_text)
    return prompt


# => Part 2.2.2: Function to run the final augmented query
def generate_final_answer(prompt):
    if not gemini_api_key:
        raise ValueError("Gemini API key not found/invalid. Please provide one!")
    # https://aws.plainenglish.io/mastering-llm-parameters-a-deep-dive-into-temperature-top-k-and-top-p-623b6aa2e6e5
    config = GenerationConfig(temperature=0.65,
                              top_p=0.9,  # Cumulative probability of next words. Lower = more predictable
                              top_k=70)  # No. of words to consider next step. Lower = More predictable
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt, generation_config=config)
    return answer.text


# => Part 2.2.3: Define function to run all tasks sequentially to get answer
def run_overall_answer_sequence(dbase, query):
    relevant_text = get_relevant_text_passage(query, dbase, n_results=4)
    print(relevant_text)
    augmented_prompt = rag_augmented_prompt_template(query, relevant_text="".join(relevant_text))
    final_answer = generate_final_answer(augmented_prompt)
    return final_answer


db = read_chroma_vector_db_collection(path='./docs/chromadb_collections', collection_name='state_of_union')
question = "Explain the sanctions that have been placed on Russia."

complete_answer = run_overall_answer_sequence(db, query=question)
print(complete_answer)


