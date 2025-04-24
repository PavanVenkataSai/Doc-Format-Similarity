import fitz
import docx2txt
from flask import Flask, request, jsonify

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from io import StringIO
import os
import nltk

nltk.download('punkt')

UPLOAD_FOLDER = 'uploads'  # Set the folder where uploaded files will be stored

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def create_rqa(doc1, doc2):
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
        return "This file is not accepted."

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

def read_pdf(filename):
    print(filename)
    context = ''

    with fitz.open(filename) as pdf_file:
        num_pages = pdf_file.page_count

        for page_num in range(num_pages):
            page = pdf_file[page_num]
            page_text = page.get_text()
            context += page_text

    return context

@app.route('/')
def home():
    return 'All is well...'

@app.route('/process-pdf', methods=['POST'])
def process_pdf():
    doc1_path = r"C:\Users\Dell\Desktop\New Req\ucl_jro_ib_template_v2_02mar2022.docx"

    # if not doc1_path:
    #     return jsonify({'error': 'Missing doc1_path in the request'})

    # Save the uploaded file to the "uploads" folder
    doc2 = request.files['doc2']
    doc2_filename = os.path.join(app.config['UPLOAD_FOLDER'], doc2.filename)
    doc2.save(doc2_filename)

    doc1_text = docx2txt.process(doc1_path)
    doc2_text = read_pdf(doc2_filename)

    result = create_rqa(split_text(doc1_text), split_text(doc2_text))
    print(result)
    return jsonify({'status': 1, 'response': result})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)  # Create the "uploads" folder if it doesn't exist
    app.run(debug=True, port=5000)