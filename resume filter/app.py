import spacy
import re
import os
import subprocess
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Ensure SpaCy model is installed and load it
def load_spacy_model():
    try:
        nlp = spacy.load('en_core_web_sm')
        return nlp
    except OSError as e:
        print("SpaCy model not found. Attempting to download...")
        subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load('en_core_web_sm')
        return nlp

# Load SpaCy model
nlp = load_spacy_model()

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return ''

# Function to extract information from a resume
def extract_info_from_resume(text):
    doc = nlp(text)
    
    def extract_info(pattern, text):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else 'Not specified'
    
    education = extract_info(r'\bEducation:\s*(.*?)\s*(?:Experience|Skills|$)', text)
    relevant_experience = extract_info(r'\bRelevant experience:\s*(.*?)\s*(?:Education|Skills|$)', text)
    years_of_experience = extract_info(r'\bYears of experience:\s*(\d+)', text)
    current_company = extract_info(r'\bCurrent company:\s*(.*?)\s*(?:Other companies|$)', text)
    other_companies = extract_info(r'\bOther companies worked for:\s*(.*?)\s*(?:Current company|$)', text)
    functional_domain = extract_info(r'\bFunctional domain worked in:\s*(.*?)\s*(?:Cities or countries|$)', text)
    cities_countries = extract_info(r'\bCities or countries worked in:\s*(.*?)\s*(?:Number of jobs|$)', text)
    number_of_jobs = extract_info(r'\bNumber of jobs:\s*(\d+)', text)
    written_communication = extract_info(r'\bWritten communication:\s*(.*?)\s*(?:Visa status|$)', text)
    visa_status = extract_info(r'\bVisa status:\s*(.*?)\s*(?:Written communication|$)', text)
    
    return {
        "Education": education,
        "Relevant Experience": relevant_experience,
        "Years of Experience": years_of_experience,
        "Current Company": current_company,
        "Other Companies Worked For": other_companies,
        "Functional Domain Worked In": functional_domain,
        "Cities or Countries Worked In": cities_countries,
        "Number of Jobs": number_of_jobs,
        "Written Communication": written_communication,
        "Visa Status": visa_status
    }

# Function to extract information from a job description
def extract_info_from_job_description(text):
    doc = nlp(text)
    skills_required = []

    skills_pattern = r'\bSkills required:\s*(.*?)\s*(?:Responsibilities|$)'
    skills_match = re.search(skills_pattern, text, re.IGNORECASE)
    if skills_match:
        skills_required = skills_match.group(1).split(',')

    return {
        "skills_required": skills_required
    }

# Function to calculate similarity between resume and job description
def calculate_similarity(resume_data, job_data):
    vectorizer = TfidfVectorizer()

    resume_text = ' '.join(resume_data.values())
    job_skills = ' '.join(job_data['skills_required'])

    vectors = vectorizer.fit_transform([resume_text, job_skills])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2]).flatten()

    return similarity[0]

# Function to rank candidates based on their fit
def rank_candidates(resumes, job_description):
    scores = []

    for resume in resumes:
        score = calculate_similarity(resume, job_description)
        scores.append(score)

    ranking = pd.DataFrame({
        'resume': resumes,
        'score': scores
    }).sort_values(by='score', ascending=False)

    return ranking

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume_file' not in request.files:
        return render_template('index.html', title='Error', result='No file part')
    
    file = request.files['resume_file']
    
    if file.filename == '':
        return render_template('index.html', title='Error', result='No selected file')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        text = extract_text_from_file(file_path)
        parsed_data = extract_info_from_resume(text)
        return render_template('index.html', title='Resume Parsing Results', result=parsed_data, result_type='Resume')
    
    return render_template('index.html', title='Error', result='Invalid file type')

@app.route('/parse_job_description', methods=['POST'])
def parse_job_description():
    text = request.form.get('job_description_text')
    parsed_data = extract_info_from_job_description(text)
    return render_template('index.html', title='Job Description Parsing Results', result=parsed_data, result_type='Job Description')

@app.route('/rank_candidates', methods=['POST'])
def rank_candidates_endpoint():
    resumes_texts = request.form.get('resumes').split('\n')
    job_description_text = request.form.get('job_description')

    resume_data = [extract_info_from_resume(resume) for resume in resumes_texts]
    job_data = extract_info_from_job_description(job_description_text)

    ranking = rank_candidates(resume_data, job_data)

    return render_template('index.html', title='Candidate Ranking Results', ranking=ranking.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
