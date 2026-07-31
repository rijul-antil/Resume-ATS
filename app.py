# app.py

import os
import re
import ssl
import nltk
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 🟢 CODE CHANGES: ROBUST NLTK DOWNLOAD & SSL FIX FOR RENDER
# Bypasses Linux SSL verification blocks and downloads required NLTK assets safely
# ==========================================
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

for resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        print(f"Warning: Could not download {resource}: {e}")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# ==========================================

app = FastAPI(title="Smart ATS Resume Copilot")

SKILL_DB = [
    "python", "javascript", "typescript", "react", "node.js", "express", "mongodb",
    "sql", "postgresql", "fastapi", "docker", "aws", "git", "machine learning",
    "nlp", "tensorflow", "pytorch", "pandas", "numpy", "html", "css", "tailwind"
]

def extract_entities(text: str):
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    phones = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    found_skills = [skill for skill in SKILL_DB if skill in text.lower()]
    return {
        "email": emails[0] if emails else "Not Found",
        "phone": phones[0] if phones else "Not Found",
        "skills": list(set(found_skills))
    }

def calculate_match_score(resume_text: str, job_text: str):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)

def find_missing_keywords(resume_text: str, job_text: str):
    job_skills = [skill for skill in SKILL_DB if skill in job_text.lower()]
    resume_skills = [skill for skill in SKILL_DB if skill in resume_text.lower()]
    return list(set(job_skills) - set(resume_skills))

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Resume & ATS Copilot</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-900 text-white min-h-screen p-8">
        <div class="max-w-4xl mx-auto">
            <h1 class="text-3xl font-bold mb-2 text-cyan-400">🤖 Smart ATS Resume Copilot</h1>
            <p class="text-slate-400 mb-8">Analyze candidate resumes against job descriptions using NLP & TF-IDF.</p>
            
            <form action="/analyze" method="post" class="space-y-6">
                <div>
                    <label class="block text-sm font-medium mb-2 text-slate-300">Paste Candidate Resume Text</label>
                    <textarea name="resume_text" rows="5" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-cyan-500" required></textarea>
                </div>
                <div>
                    <label class="block text-sm font-medium mb-2 text-slate-300">Paste Job Description</label>
                    <textarea name="job_text" rows="5" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-cyan-500" required></textarea>
                </div>
                <button type="submit" class="w-full bg-cyan-500 hover:bg-cyan-600 font-semibold py-3 px-6 rounded-lg transition duration-200">Run ATS Analysis</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/analyze", response_class=HTMLResponse)
def analyze(resume_text: str = Form(...), job_text: str = Form(...)):
    entities = extract_entities(resume_text)
    score = calculate_match_score(resume_text, job_text)
    missing_skills = find_missing_keywords(resume_text, job_text)

    skills_badge = "".join([f'<span class="bg-cyan-900 text-cyan-300 px-3 py-1 rounded-full text-xs font-semibold mr-2">{s}</span>' for s in entities["skills"]])
    missing_badge = "".join([f'<span class="bg-rose-900 text-rose-300 px-3 py-1 rounded-full text-xs font-semibold mr-2">{s}</span>' for s in missing_skills])

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ATS Results</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-900 text-white min-h-screen p-8">
        <div class="max-w-4xl mx-auto space-y-6">
            <a href="/" class="text-cyan-400 hover:underline">← Analyze Another Resume</a>
            
            <div class="bg-slate-800 border border-slate-700 rounded-lg p-6">
                <h2 class="text-2xl font-bold text-cyan-400 mb-4">ATS Match Score: {score}%</h2>
                <div class="w-full bg-slate-700 h-4 rounded-full overflow-hidden">
                    <div class="bg-cyan-500 h-full" style="width: {score}%"></div>
                </div>
            </div>

            <div class="grid grid-cols-2 gap-6">
                <div class="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-3">
                    <h3 class="text-lg font-bold text-slate-200">Candidate Information</h3>
                    <p class="text-sm text-slate-400"><strong>Email:</strong> {entities['email']}</p>
                    <p class="text-sm text-slate-400"><strong>Phone:</strong> {entities['phone']}</p>
                    <div class="pt-2">
                        <p class="text-sm text-slate-400 mb-2"><strong>Extracted Skills:</strong></p>
                        <div class="flex flex-wrap gap-2">{skills_badge or '<span class="text-slate-500">None detected</span>'}</div>
                    </div>
                </div>

                <div class="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-3">
                    <h3 class="text-lg font-bold text-rose-400">Missing Required Skills</h3>
                    <p class="text-xs text-slate-400">Skills in JD missing from Resume:</p>
                    <div class="flex flex-wrap gap-2 pt-2">{missing_badge or '<span class="text-emerald-400 text-sm">None! All matching.</span>'}</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

# ==========================================
# 🟢 CODE CHANGES: RENDER PRODUCTION ENTRYPOINT
# Directly binds the app instance to host 0.0.0.0 and reads Render's assigned $PORT env var
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
# ==========================================
