import fitz
import docx2txt
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from io import StringIO
import os
import nltk

UPLOAD_FOLDER = 'uploaded_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# nltk.download('punkt')

st.title("Document Format Checker")

# Create file uploader
# docx_file = st.file_uploader("Upload a DOCX file")
pdf_file = st.file_uploader("Upload a PDF file")

def create_rqa(doc1, doc2):
    # Join the tokens to form strings for TF-IDF
    doc1_str = ' '.join(doc1)
    doc2_str = ' '.join(doc2)

    # Use TF-IDF and cosine similarity to calculate document similarity
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([doc1_str, doc2_str])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    # Define a threshold for similarity, you can adjust this as needed
    similarity_threshold = 0.7  # Adjust as needed

    if similarity_score > similarity_threshold:
        return "This file is accepted."
    else:
        return "This file is not accepted"

# Function to split text
def split_text(text, chunk_size=2000):
    """
    Splits the given text into chunks of approximately the specified chunk size.

    Args:
    text (str): The text to split.

    chunk_size (int): The desired size of each chunk (in characters).

    Returns:
    List[str]: A list of chunks, each of approximately the specified chunk size.
    """

    chunks = []
    current_chunk = StringIO()
    current_size = 0
    sentences = sent_tokenize(text)
    for sentence in sentences:
        sentence_size = len(sentence)
        if sentence_size > chunk_size:
            while sentence_size > chunk_size:
                chunk = sentence[:chunk_size]
                chunks.append(chunk)
                sentence = sentence[chunk_size:]
                sentence_size -= chunk_size
                current_chunk = StringIO()
                current_size = 0
        if current_size + sentence_size < chunk_size:
            current_chunk.write(sentence)
            current_size += sentence_size
        else:
            chunks.append(current_chunk.getvalue())
            current_chunk = StringIO()
            current_chunk.write(sentence)
            current_size = sentence_size
    if current_chunk:
        chunks.append(current_chunk.getvalue())
    return chunks

# Function to read PDF
def read_pdf(file):
    context = ''
    with fitz.open(file) as pdf_file:
        num_pages = pdf_file.page_count
        for page_num in range(num_pages):
            page = pdf_file[page_num]
            page_text = page.get_text()
            context += page_text
    return context

if pdf_file:
    # Save the uploaded files to the specified folder
    docx_file_path = r"C:\Users\Dell\Desktop\New Req\ucl_jro_ib_template_v2_02mar2022.docx"
    pdf_file_path = os.path.join(UPLOAD_FOLDER, pdf_file.name)

    with open(pdf_file_path, 'wb') as f:
        f.write(pdf_file.read())

    # Read the saved files
    doc1_text = docx2txt.process(docx_file_path)
    doc2_text = read_pdf(pdf_file_path)

    # Calculate similarity
    result = create_rqa(split_text(doc1_text), split_text(doc2_text))
    st.write("--> Result:", result)