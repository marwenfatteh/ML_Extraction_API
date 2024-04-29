import json
import re
import PyPDF2
import pyodbc
from flask import Flask, request, jsonify
from docquery import document, pipeline
import pandas as pd
import spacy
from flair.data import Sentence
from flair.models import SequenceTagger
import tempfile
import os


app = Flask(__name__)

nlp = spacy.load('en_core_web_sm')


def connect_to_database():
    try:
        connection = pyodbc.connect(
            'DRIVER={SQL Server};'
            'SERVER=Marwen\SQLEXPRESS;'
            'DATABASE=identity_login_register;'
        )
        return connection
    except Exception as e:
        print("Error connecting to database:", e)
        return None



# Function to save to database
def save_to_database(email,client,team,tech,proj,refId):
    connection = connect_to_database()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("INSERT INTO ExportedReferences (Project, Client, Tech, Contact, Team, referenceDocumentId) VALUES (?, ?, ?, ?, ?, ?);", (proj,client,tech,email,team,refId))
            connection.commit()
            cursor.close()
            connection.close()
            print("saved to database successfully")
        except Exception as e:
            print("Error saving to database:", e)
    else:
        print("Connection to database failed")

tagger = SequenceTagger.load('ner')
# Standardized question
standard_question = "What is the client?"

# Function to process the PDF file and return the answer
def process_pdf(pdf_file):
    # Save the uploaded PDF file to a temporary directory
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, 'uploaded_file.pdf')
    pdf_file.save(pdf_path)
    txt = extract_text_from_pdf(pdf_path)
    email = extract_emails(txt)
    nlp1 = spacy.load('en_core_web_sm')

    sentence = Sentence(txt)
    # Run NER on the sentence
    tagger.predict(sentence)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    arabic_names = ""
    for entity in sentence.get_spans('ner'):
        if entity.tag == 'PER' and any(char.isalpha() for char in entity.text):
            arabic_names = arabic_names + entity.text + ","
    # Load the document and create the pipeline
    doc = document.load_document(pdf_path)
    p = pipeline('document-question-answering')

    nlp_text = nlp1(txt)
    noun_chunks = nlp_text.noun_chunks
    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]

    # reading the csv file
    data = pd.read_csv(r'C:\Users\fateh\PycharmProjects\Skill_Extraction\res\skills (1).csv')

    # extract values
    skills = data.columns.values.tolist()
    skillset = ""

    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills:

            skillset = skillset+token+","

    # check for bi-grams and tri-grams (example: machine learning)
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset = skillset+token+","




    # Process the standardized question using the pipeline and document context
    result = p(question=standard_question, **doc.context)
    client = result[0]['answer']

    # Delete the temporary directory and its contents
    os.remove(pdf_path)
    os.rmdir(temp_dir)


    return [client,skillset,email,arabic_names]

# Function to extract emails from text
def extract_emails(text):
    # Regular expression for extracting emails
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for email in re.findall(email_regex, text):
        return email

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


@app.route('/ask', methods=['POST'])
def ask_question():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    pdf_file = request.files['file']
    proj = request.form .get('Project')
    refId = request.form .get('Id')
    print(proj)
    print(refId)
    # Check if the file is of PDF type
    if pdf_file.filename == '' or not pdf_file.filename.endswith('.pdf'):
        return jsonify({'error': 'Invalid file format. Please upload a PDF file.'})

    # Process the PDF file
    answer = process_pdf(pdf_file)

    jsret = json.dumps(answer)
    data = json.loads(jsret)
    print(data[2],data[1],data[0],data[3],proj,refId)

    save_to_database(data[2],data[0],data[3],data[1],proj,refId)
    # Return the answer as JSON response
    return jsret

if __name__ == '__main__':
    app.run(debug=True)
