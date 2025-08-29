# ====== app/core/cv_processor.py (UPDATED VERSION) ======
import asyncio
import os
import PyPDF2
import pdfplumber
import json
import re
from typing import Dict, Optional, List, Tuple
from config import settings
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.stem import PorterStemmer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model (optimized for performance)
# try:
#     nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
# except Exception as e:
#     logger.error(f"Failed to load spaCy model: {e}")
#     raise

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

# Define job requirements for RAN Engineer/Wireless Engineer
JOB_SPEC = {
    "role": "RAN Engineer/Wireless Engineer",
    "required_skills": [
        "First and second level faulty handling support of 2G/3G/4G LTE",
        "Commissioning and integration of BTS, NodeB, eNodeB, LTE & gNodeB",
        "eNodeB/nodeB/BTS survey and installation",
        "KPI analysis fault troubleshooting",
        "Work with cross domain teams",
        "Understanding of network Topology",
        "Analysis of health check reports",
        "Software patches/software upgrade as per OEM Suggestions"
    ]
}


# Helper function to safely delete files
def safe_delete_file(file_path: str) -> None:
    """
    Safely delete a file and log the result.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        else:
            print(f"File not found for deletion: {file_path}")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

async def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using multiple methods for best results
    """
    print(f"📄 Extracting text from: {os.path.basename(pdf_path)}")
    
    text = ""
    
    # Method 1: Try pdfplumber first (better for complex layouts)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if text.strip():
            print(f"   ✅ Successfully extracted text using pdfplumber ({len(text)} characters)")
            return text
    except Exception as e:
        print(f"   ⚠️ pdfplumber failed: {e}")
    
    # Method 2: Fallback to PyPDF2
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text: 
                    text += page_text + "\n"
        
        if text.strip():
            print(f"   ✅ Successfully extracted text using PyPDF2 ({len(text)} characters)")
            return text
    except Exception as e:
        print(f"   ❌ PyPDF2 also failed: {e}")
    
    print(f"   ❌ Failed to extract text from PDF")
    return ""

def calculate_job_similarity(cv_text: str, job_spec: Dict) -> Tuple[float, List[str]]:
    """
    Calculate similarity between CV and job requirements using TF-IDF and spaCy
    Returns: (similarity_score, matched_skills)
    """
    # try:
    #     # Preprocess texts
    #     cv_doc = nlp(cv_text.lower())
    #     print(f"   Job Sppec: {job_spec}")
    #     job_doc = nlp(" ".join(job_spec["required_skills"]).lower())
        
    #     # Extract tokens and remove stop words
    #     cv_tokens = [token.lemma_ for token in cv_doc if not token.is_stop and token.is_alpha]
    #     job_tokens = [token.lemma_ for token in job_doc if not token.is_stop and token.is_alpha]
        
    #     # Create TF-IDF vectors
    #     # vectorizer = TfidfVectorizer()
    #     vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    #     tfidf_matrix = vectorizer.fit_transform([
    #         " ".join(cv_tokens),
    #         " ".join(job_tokens)
    #     ])
        
    #     # Calculate cosine similarity
    #     similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
    #     # Find matched skills
    #     matched_skills = []
    #     cv_text_lemmas = set(token.lemma_ for token in cv_doc if not token.is_stop and token.is_alpha)
    #     for skill in job_spec["required_skills"]:
    #         # Use spaCy for semantic matching
    #         skill_doc = nlp(skill.lower())
    #         skill_lemmas = set(token.lemma_ for token in skill_doc if not token.is_stop)
    #         if any(lemma in cv_text_lemmas for lemma in skill_lemmas):
    #             matched_skills.append(skill)
        
    #     return round(similarity, 2), matched_skills
    # except Exception as e:
    #     logger.error(f"Error calculating job similarity: {e}")
    #     return 0.0, []

    # try:
    #     # Preprocess texts by joining skills and converting to lowercase
    #     job_skills_text = " ".join(job_spec.get("required_skills", [])).lower()
    #     cv_text_lower = cv_text.lower()

    #     # Create TF-IDF vectors
    #     # TfidfVectorizer handles tokenization and stop word removal automatically
    #     vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    #     tfidf_matrix = vectorizer.fit_transform([cv_text_lower, job_skills_text])
        
    #     # Calculate cosine similarity
    #     similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    #     # Find matched skills based on simple string matching
    #     matched_skills = []
    #     cv_words = set(cv_text_lower.split())
        
    #     for skill in job_spec.get("required_skills", []):
    #         if skill.lower() in cv_words:
    #             matched_skills.append(skill)
                
    #     return round(similarity, 2), matched_skills
    # except Exception as e:
    #     print(f"Error calculating job similarity: {e}")
    #     return 0.0, []

    # try:
    #     # Preprocess texts
    #     cv_doc = nlp(cv_text.lower())
    #     job_doc = nlp(" ".join(job_spec["required_skills"]).lower())

    #     # Extract tokens and apply stemming
    #     cv_tokens = [stemmer.stem(token.text) for token in cv_doc if not token.is_stop and token.is_alpha]
    #     job_tokens = [stemmer.stem(token.text) for token in job_doc if not token.is_stop and token.is_alpha]
        
    #     # Create TF-IDF vectors
    #     vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    #     tfidf_matrix = vectorizer.fit_transform([
    #         " ".join(cv_tokens),
    #         " ".join(job_tokens)
    #     ])
        
    #     # Calculate cosine similarity
    #     similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
    #     # Find matched skills
    #     matched_skills = []
    #     cv_text_stems = set(stemmer.stem(token.text) for token in cv_doc if not token.is_stop and token.is_alpha)
        
    #     for skill in job_spec["required_skills"]:
    #         skill_doc = nlp(skill.lower())
    #         skill_stems = set(stemmer.stem(token.text) for token in skill_doc if not token.is_stop)
            
    #         if any(stem in cv_text_stems for stem in skill_stems):
    #             matched_skills.append(skill)
        
    #     return round(similarity, 2), matched_skills
    # except Exception as e:
    #     # It's better to use a logger here
    #     print(f"Error calculating job similarity: {e}")
    #     return 0.0, []

    try:
        # --- Part 1: TF-IDF Calculation ---
        # Preprocess texts
        cv_doc = nlp(cv_text.lower())
        job_doc = nlp(" ".join(job_spec["required_skills"]).lower())

        # Extract tokens and apply stemming
        cv_tokens = [stemmer.stem(token.text) for token in cv_doc if not token.is_stop and token.is_alpha]
        job_tokens = [stemmer.stem(token.text) for token in job_doc if not token.is_stop and token.is_alpha]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        tfidf_matrix = vectorizer.fit_transform([
            " ".join(cv_tokens),
            " ".join(job_tokens)
        ])
        
        # Calculate cosine similarity (our first component of the score)
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # --- Part 2: Find Matched Skills and Calculate a Percentage Score ---
        matched_skills = []
        cv_text_stems = set(stemmer.stem(token.text) for token in cv_doc if not token.is_stop and token.is_alpha)
        
        for skill in job_spec["required_skills"]:
            skill_doc = nlp(skill.lower())
            skill_stems = set(stemmer.stem(token.text) for token in skill_doc if not token.is_stop)
            
            if any(stem in cv_text_stems for stem in skill_stems):
                matched_skills.append(skill)
        
        # Calculate the percentage of matched skills
        total_required_skills = len(job_spec.get("required_skills", []))
        if total_required_skills > 0:
            skill_match_percentage = len(matched_skills) / total_required_skills
        else:
            skill_match_percentage = 0.0

        # --- Part 3: Combine Scores with a Weighted Average ---
        # We'll give more weight to the skill count since you want it to have a bigger impact.
        # You can adjust these weights (e.g., 0.6 and 0.4) as needed.
        final_score = (tfidf_score * 0.4) + (skill_match_percentage * 0.6)
        
        return round(final_score, 2), matched_skills
    except Exception as e:
        print(f"Error calculating job similarity: {e}")
        return 0.0, []


def parse_cv_text_to_json_ultra_robust(text: str, name: str = "",  job_spec: Dict = None) -> Dict:
    """
    Ultra-robust CV parser trained on 8+ diverse LinkedIn CV formats
    Handles Nigerian, US, international formats with advanced pattern matching
    """
    print(f"🔍 Parsing CV text for: {name}")
    print(f"   Job Sppec: {job_spec}")
    # Initialize the result structure
    cv_data = {
        "Name": name,
        "Location": "",
        "Contact_Information": {
            "Email": "",
            "Phone": "",
            "LinkedIn": "",
            "Website": "",
            "Portfolio": ""
        },
        "Top_Skills": [],
        "Certifications": [],
        "Honors_Awards": [],
        "Summary": "",
        "Experience": [],
        "Education": [],
        "Languages": [],
        "Job_Match": {
            "role":  job_spec.get("role", "") if job_spec else "",
            "similarity_score": 0.0,
            "matched_skills": []
        }
    }
    
    # Clean and normalize text - preserve structure for parsing
    original_text = text
    text_with_breaks = re.sub(r'\n\s*\n', '\n', text).replace('\r', '')
    print("\n******1111*************\n")
    text_single_line = re.sub(r'\s+', ' ', text).strip()
    print(f"\n******1111222*************\n, {name}")

    # Calculate job similarity
    print(f"   Job Sppec: {job_spec}")
    similarity_score, matched_skills = calculate_job_similarity(text,  job_spec or {})
    cv_data["Job_Match"]["similarity_score"] = similarity_score
    cv_data["Job_Match"]["matched_skills"] = matched_skills

    try:
        # 1. EXTRACT NAME (Multiple sophisticated patterns)
        if not name or name == "":
            name_patterns = [
                # Pattern 1: Name followed by professional title
                r'^([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\n(?:.*?(?:Engineer|Developer|Analyst|Manager|Specialist))',
                # Pattern 2: Name before location on separate lines
                r'([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\n(?:[A-Za-z\s,]+(?:Nigeria|United States|California))',
                # Pattern 3: Name after contact section
                r'Contact\n.*?\n.*?\n.*?\n([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                # Pattern 4: Standalone name pattern
                r'\n([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\n(?:[A-Z][a-z].*?(?:Engineer|Developer|Analyst))',
                # Pattern 5: Name at document start
                r'^([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
            ]
            
            for pattern in name_patterns:
                print("\n******111333*************\n")
                match = re.search(pattern, text_with_breaks, re.MULTILINE)
                print("\n******1111444*************\n")
                if match:
                    print("\n******111555*************\n")
                    potential_name = match.group(1).strip()
                    print("\n******1116666*************\n")
                    # Validate it's actually a name (not a company or skill)
                    if not re.search(r'(?:University|School|Company|Technologies|Services|Engineering|Development)', potential_name, re.IGNORECASE):
                        cv_data["Name"] = potential_name
                        break
        
        # 2. EXTRACT CONTACT INFORMATION
        
        # Email - Enhanced patterns
        email_patterns = [
            r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
            r'(?:Email|E-mail)[\s:]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            r'Contact\s*\n(?:[^\n]*\n)*?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        ]
        for pattern in email_patterns:
            print("\n******11177*************\n")
            emails = re.findall(pattern, text_single_line, re.IGNORECASE)
            print("\n******111888*************\n")
            if emails:
                print("\n******111999*************\n")
                email = emails[0] if isinstance(emails[0], str) else emails[0]
                # Filter out generic/company emails
                if not re.search(r'@(?:company|example|test|noreply)', email.lower()):
                    cv_data["Contact_Information"]["Email"] = email
                    break
        
        # Phone - Ultra-comprehensive patterns
        phone_patterns = [
            r'\+234\d{10}',  # Nigerian international format
            r'0\d{10}',      # Nigerian local format  
            r'08\d{9}',      # Nigerian mobile format
            r'\+\d{1,4}[\s\-]?\d{7,14}',  # International format
            r'\(\+\d{1,4}\)[\s\-]?\d{7,14}',  # International with parentheses
            r'\d{4}[\s\-]?\d{3}[\s\-]?\d{4}',  # US format with separators
            r'\d{11}(?:\s*\([^)]*\))?',  # 11 digits with optional notes
            r'\d{10}(?:\s*\([^)]*\))?'   # 10 digits with optional notes
        ]
        for pattern in phone_patterns:
            print("\n******111-----1*************\n")
            phones = re.findall(pattern, text_single_line)
            print("\n******111-----2*************\n")
            if phones:
                phone = phones[0].strip()
                # Clean up phone number
                phone = re.sub(r'\s*\([^)]*\)$', '', phone)  # Remove trailing notes
                cv_data["Contact_Information"]["Phone"] = phone
                break
        
        # LinkedIn - Enhanced patterns for various formats
        linkedin_patterns = [
            r'(?:linkedin\.com/in/|www\.linkedin\.com/in/)([a-zA-Z0-9\-]+)',
            r'LinkedIn[)\s]*([a-zA-Z0-9\-\.]+)',
            r'www\.linkedin\.com/in/([a-zA-Z0-9\-]+)',
            r'linkedin\.com/in/([a-zA-Z0-9\-]+)',
            r'\(LinkedIn\)\s*\n?([a-zA-Z0-9\-\.]+)'
        ]
        print("\n******111-----52828255*************\n")
        for pattern in linkedin_patterns:
            linkedin_matches = re.findall(pattern, text_single_line, re.IGNORECASE)
            if linkedin_matches:
                username = linkedin_matches[0]
                cv_data["Contact_Information"]["LinkedIn"] = f"https://www.linkedin.com/in/{username}"
                break
        
        # Portfolio/Website - Multiple patterns
        portfolio_patterns = [
            r'(?:Portfolio|Website|Personal)[\s)]*\n?([a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:/[a-zA-Z0-9\-]*)?)',
            r'(linktr\.ee/[a-zA-Z0-9\-]+)',
            r'([a-zA-Z0-9\-]+\.(?:vercel\.app|netlify\.app|github\.io)/?[a-zA-Z0-9\-/]*)',
            r'((?:https?://)?[a-zA-Z0-9\-\.]+\.(?:app|dev|io|com)/[a-zA-Z0-9\-]*)',
            r'\(Personal\)\s*\n?([a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})'
        ]
        print("\n******111-----555282828*************\n")
        for pattern in portfolio_patterns:
            
            portfolio_matches = re.findall(pattern, text_single_line, re.IGNORECASE)
            if portfolio_matches:
                portfolio = portfolio_matches[0]
                if not portfolio.startswith('http'):
                    portfolio = 'https://' + portfolio
                cv_data["Contact_Information"]["Portfolio"] = portfolio
                break
        
        # 3. EXTRACT LOCATION - Enhanced for international formats
        location_patterns = [
            # Nigerian format: Area, State, Nigeria
            r'([A-Za-z\s]+,\s*Lagos\s*State,\s*Nigeria)',
            r'([A-Za-z\s]+,\s*[A-Za-z\s]+\s*State,\s*Nigeria)',
            # US format: City, State, Country
            r'([A-Za-z\s]+,\s*California,\s*United States)',
            r'([A-Za-z\s]+,\s*[A-Za-z\s]+,\s*United States)',
            # General format: City, Country
            r'([A-Za-z\s]+,\s*Nigeria)',
            r'([A-Za-z\s]+,\s*United States)',
            # After Contact section
            r'Contact\s*\n([^\n]+(?:Nigeria|United States|California)[^\n]*)',
            # Before Summary or after Name
            r'(?:Name|Summary)\s*\n(?:[^\n]*\n)?([A-Za-z\s,]+(?:Nigeria|United States|California)[^\n]*)',
            # Standalone location line
            r'^([A-Za-z\s,]+(?:Nigeria|United States|California))\.?$'
        ]
        print("\n******111-----55500000*************\n")
        for pattern in location_patterns:
            locations = re.findall(pattern, text_with_breaks, re.MULTILINE | re.IGNORECASE)
            if locations:
                location = locations[0].strip().rstrip('.')
                # Clean up location
                location = re.sub(r'^[^A-Za-z]*', '', location)  # Remove leading non-letters
                cv_data["Location"] = location
                break
        
        # 4. EXTRACT TOP SKILLS - Enhanced for various formats
        skills_section_patterns = [
            # Standard "Top Skills" section
            r'Top Skills\s*\n((?:[^\n]+\n?)+?)(?=\n(?:Languages|Certifications|Experience|Education|Summary|Contact|$))',
            # Alternative skill section headers
            r'(?:Skills|Technical Skills|Key Skills|Core Skills)[\s:]*\n?((?:[^\n]+\n?)+?)(?=\n(?:Languages|Certifications|Experience|Education|$))',
            # Skills in summary section
            r'(?:Technologies|Skills):\s*([A-Za-z0-9\s,\.+#-]+(?:AWS|Python|JavaScript|React|Node|SQL)[A-Za-z0-9\s,\.+#-]*)',
            # Skills in parentheses or after colon
            r'Languages:\s*([A-Za-z0-9\s,+#-]+)'
        ]
        print("\n******111-----888888*************\n")
        for pattern in skills_section_patterns:
            skills_match = re.search(pattern, text_with_breaks, re.IGNORECASE | re.MULTILINE)
            if skills_match:
                skills_text = skills_match.group(1).strip()
                
                # Handle different skill formats
                if '\n' in skills_text:
                    # Line-separated skills
                    skills_list = [skill.strip() for skill in skills_text.split('\n') if skill.strip()]
                else:
                    # Comma-separated skills
                    skills_list = [skill.strip() for skill in re.split(r'[,;]', skills_text) if skill.strip()]
                
                # Filter and clean skills
                cleaned_skills = []
                for skill in skills_list[:15]:  # Limit to 15 skills
                    skill = skill.strip()
                    if skill and len(skill) > 1 and not re.match(r'^[^A-Za-z]*$', skill):
                        cleaned_skills.append(skill)
                
                if cleaned_skills:
                    cv_data["Top_Skills"] = cleaned_skills
                    break
        
        # 5. EXTRACT LANGUAGES (New section)
        language_patterns = [
            r'Languages\s*\n((?:[^\n]+\n?)+?)(?=\n(?:Certifications|Experience|Education|Top Skills|$))',
            r'(?:Language Skills|Spoken Languages)[\s:]*\n?((?:[^\n]+\n?)+?)(?=\n[A-Z][a-z]+|$)'
        ]
        print("\n******111-----5777755*************\n")
        for pattern in language_patterns:
            lang_match = re.search(pattern, text_with_breaks, re.IGNORECASE | re.MULTILINE)
            if lang_match:
                lang_text = lang_match.group(1).strip()
                lang_list = [lang.strip() for lang in lang_text.split('\n') if lang.strip()]
                cv_data["Languages"] = lang_list[:10]
                break
        
        # 6. EXTRACT CERTIFICATIONS - Enhanced patterns
        cert_section_patterns = [
            r'Certifications\s*\n((?:[^\n]+\n?)+?)(?=\n(?:Honors|Languages|Experience|Education|Summary|$))',
            r'(?:Certificates?|Professional Certifications|Licenses)[\s:]*\n?((?:[^\n]+\n?)+?)(?=\n(?:Honors|Languages|Experience|Education|$))'
        ]

        print("\n******111-----55333*************\n")
        for pattern in cert_section_patterns:
            cert_match = re.search(pattern, text_with_breaks, re.IGNORECASE | re.MULTILINE)
            if cert_match:
                cert_text = cert_match.group(1).strip()
                cert_list = [cert.strip() for cert in cert_text.split('\n') if cert.strip() and len(cert.strip()) > 3]
                cv_data["Certifications"] = cert_list[:15]
                break
        
        # 7. EXTRACT HONORS & AWARDS
        awards_patterns = [
            r'(?:Honors-Awards|Honors|Awards?|Achievements?|Recognition)\s*\n((?:[^\n]+\n?)+?)(?=\n(?:Experience|Education|Languages|$))',
            r'(?:Awards and Recognition|Achievements and Awards)[\s:]*\n?((?:[^\n]+\n?)+?)(?=\n[A-Z][a-z]+|$)'
        ]

        print("\n******111-----5559999*************\n")
        for pattern in awards_patterns:
            awards_match = re.search(pattern, text_with_breaks, re.IGNORECASE | re.MULTILINE)
            if awards_match:
                awards_text = awards_match.group(1).strip()
                awards_list = [award.strip() for award in awards_text.split('\n') if award.strip() and len(award.strip()) > 5]
                cv_data["Honors_Awards"] = awards_list[:10]
                break
        
        # 8. EXTRACT SUMMARY - Enhanced patterns
        summary_patterns = [
            r'Summary\s*\n([^\n].{20,500}?)\n(?=Experience|Education|Skills)',
            r'(?:Professional Summary|About|Profile)[\s:]*\n?([^\n].{20,500}?)\n(?=Experience|Education)',
            r'Summary\s*:?\s*([^\n]+(?:\n[^\n]+){0,3})'
        ]
        print("\n******111-----55444*************\n")
        for pattern in summary_patterns:
            summary_match = re.search(pattern, text_with_breaks, re.IGNORECASE | re.MULTILINE)
            if summary_match:
                summary = summary_match.group(1).strip()
                # Clean up summary - remove excessive whitespace
                summary = re.sub(r'\s+', ' ', summary)
                cv_data["Summary"] = summary
                break
        
        # 9. EXTRACT EXPERIENCE - Ultra-sophisticated parsing
        experience_patterns = [
            r'Experience\s*\n(.*?)(?=\nEducation|\nSkills|\nCertifications|\nLanguages|Page\s+\d+|$)',
        ]
        print("\n******111-----555*************\n")
        for pattern in experience_patterns:
            exp_match = re.search(pattern, text_with_breaks, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if exp_match:
                exp_text = exp_match.group(1).strip()
                exp_entries = []
                
                # Split by company patterns - more sophisticated
                # Look for lines that are likely company names
                lines = exp_text.split('\n')
                current_company = ""
                current_role = ""
                current_duration = ""
                current_location = ""
                current_description = []
                
                i = 0
                while i < len(lines):
                    print('\n\n')
                    print(len(lines))
                    print(i)
                    print('\n\n')
                    line = lines[i].strip()
                    if not line:
                        i += 1
                        continue
                    
                    # Check if line is a company name (usually standalone, capitalized)
                    if (re.match(r'^[A-Z][A-Za-z\s&._-]+$', line) and 
                        not re.search(r'(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{4})', line) and
                        not re.search(r'^\d+\s+(?:year|month)', line) and
                        len(line) > 3):
                        
                        # Save previous entry if exists
                        if current_company:
                            entry = format_experience_entry(current_company, current_role, current_duration, current_location, current_description)
                            if entry:
                                exp_entries.append(entry)
                        
                        # Start new company
                        current_company = line
                        current_role = ""
                        current_duration = ""
                        current_location = ""
                        current_description = []
                    
                    # Check if line contains dates (likely a role with duration)
                    elif re.search(r'(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2})\s+\d{4}', line):
                        current_duration = line
                    
                    # Check if line is a location (contains state/country)
                    elif re.search(r'(?:State|Nigeria|United States|California)', line, re.IGNORECASE):
                        current_location = line
                    
                    # Check if line is a role title (not a description)
                    elif (not re.search(r'^[•\-]', line) and 
                          not current_role and 
                          len(line) < 100 and
                          not re.search(r'(?:Improved|Developed|Implemented|Led|Managed|Created)', line)):
                        current_role = line
                    
                    # Otherwise, it's likely a description
                    else:
                        if line.startswith('•') or line.startswith('-') or len(line) > 50:
                            current_description.append(line)
                    
                    i += 1
                
                # Don't forget the last entry
                if current_company:
                    entry = format_experience_entry(current_company, current_role, current_duration, current_location, current_description)
                    if entry:
                        exp_entries.append(entry)
                
                cv_data["Experience"] = exp_entries[:8]  # Limit to 8 most recent
                break
        
        # 10. EXTRACT EDUCATION - Enhanced for various formats
        education_patterns = [
            r'Education\s*\n(.*?)(?=\nExperience|\nSkills|\nCertifications|\nLanguages|Page\s+\d+|$)',
        ]
        
        for pattern in education_patterns:
            edu_match = re.search(pattern, text_with_breaks, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if edu_match:
                edu_text = edu_match.group(1).strip()
                edu_entries = []
                
                # Parse education entries
                lines = [line.strip() for line in edu_text.split('\n') if line.strip()]
                
                current_entry = {}
                for line in lines:
                    # Check if it's an institution
                    if re.search(r'(?:University|School|College|Institute|Academy)', line, re.IGNORECASE):
                        if current_entry:
                            edu_entries.append(format_education_entry(current_entry))
                        current_entry = {"institution": line}
                    
                    # Check if it's a degree
                    elif re.search(r'(?:Bachelor|Master|PhD|Diploma|Certificate|Degree|BS|MS|BSc|MSc)', line, re.IGNORECASE):
                        current_entry['degree'] = line
                    
                    # Check if it contains dates
                    elif re.search(r'\d{4}', line):
                        current_entry['duration'] = line
                    
                    # Check if it's a field of study
                    elif re.search(r'(?:Engineering|Science|Arts|Business|Computer|Software)', line, re.IGNORECASE):
                        if 'degree' not in current_entry:
                            current_entry['field'] = line
                
                if current_entry:
                    edu_entries.append(format_education_entry(current_entry))
                
                cv_data["Education"] = edu_entries
                break
        
        print(f"   ✅ Successfully parsed CV data with {len(cv_data['Experience'])} experience entries and {len(cv_data['Education'])} education entries")
        return cv_data
        
    except Exception as e:
        print(f"   ❌ Error parsing CV text: {e}")
        import traceback
        traceback.print_exc()
        return cv_data

def format_experience_entry(company, role, duration, location, description):
    """Format experience entry into a readable string"""
    if not company:
        return None
    
    entry = company
    if role:
        entry += f" - {role}"
    if duration:
        entry += f" ({duration})"
    if location:
        entry += f" | {location}"
    if description:
        # Take first 2 description points
        desc_text = ". ".join(description[:2])
        if desc_text:
            entry += f": {desc_text}"
    
    return entry

def format_education_entry(entry_dict):
    """Format education entry into a readable string"""
    parts = []
    if 'degree' in entry_dict:
        parts.append(entry_dict['degree'])
    elif 'field' in entry_dict:
        parts.append(entry_dict['field'])
    
    if 'institution' in entry_dict:
        parts.append(entry_dict['institution'])
    
    if 'duration' in entry_dict:
        parts.append(f"({entry_dict['duration']})")
    
    return " - ".join(parts) if parts else "Unknown Education"

async def process_downloaded_cv(pdf_path: str, profile_name: str, job_spec:dict) -> Optional[Dict]:
    print(f"   Job Sppec: {job_spec}")
    """
    Process a downloaded CV: extract text, parse to JSON, save files
    """
    print(f"\n📋 Processing CV for: {profile_name}")
    
    try:
        # Step 1: Extract text from PDF
        cv_text = await extract_text_from_pdf(pdf_path)
        
        if not cv_text.strip():
            print(f"   ❌ No text extracted from PDF")
            return None
        safe_delete_file(pdf_path)  # Delete PDF
        # Step 2: Save text file
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        text_file_path = os.path.join(settings.PROCESSED_PATH, f"{base_name}.txt")
        
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(cv_text)
        print(f"   💾 Text saved to: {text_file_path}")
        
        # Step 3: Parse text to structured JSON using ULTRA-ROBUST parser
        print(f"   Job Sppec: {job_spec}")
        cv_json = parse_cv_text_to_json_ultra_robust(cv_text, profile_name, job_spec)
        
        # Step 4: Save JSON file
        json_file_path = os.path.join(settings.PROCESSED_PATH, f"{base_name}.json")
        
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(cv_json, f, indent=2, ensure_ascii=False)
        print(f"   💾 JSON saved to: {json_file_path}")
        
        # Step 5: Display summary
        print(f"   📊 Extracted Information Summary:")
        print(f"      - Name: {cv_json['Name']}")
        print(f"      - Location: {cv_json['Location']}")
        print(f"      - Email: {cv_json['Contact_Information']['Email']}")
        print(f"      - Skills: {len(cv_json['Top_Skills'])} found")
        print(f"      - Experience entries: {len(cv_json['Experience'])}")
        print(f"      - Education entries: {len(cv_json['Education'])}")
        print(f"      - Certifications: {len(cv_json['Certifications'])}")
        print(f"      - Languages: {len(cv_json['Languages'])} found")
        
        
        # Step 6: Clean up files (PDF, TXT, JSON, and screenshot)
        print(f"   🗑️ Cleaning up files for {base_name}")
        safe_delete_file(text_file_path)  # Delete TXT
        safe_delete_file(json_file_path)  # Delete JSON
        
        # Attempt to delete screenshot (assuming .png or .jpg extension)
        screenshot_png_path = os.path.join(settings.SCREENSHOTS_PATH, f"{base_name}.png")
        safe_delete_file(screenshot_png_path)  # Delete PNG screenshot if exists
        
        print(f"   ✅ Cleanup completed for {base_name}")
        
        return cv_json
        
    except Exception as e:
        print(f"   ❌ Error processing CV: {e}")
        return None

# ====== UPDATED app/models/schemas.py ======
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class ContactInfo(BaseModel):
    email: Optional[str] = ""
    phone: Optional[str] = ""
    linkedin: Optional[str] = ""
    website: Optional[str] = ""
    portfolio: Optional[str] = ""  # NEW FIELD

class ProfileData(BaseModel):
    name: str
    location: Optional[str] = ""
    contact_information: ContactInfo
    top_skills: List[str] = []
    certifications: List[str] = []
    honors_awards: List[str] = []
    summary: Optional[str] = ""
    experience: List[str] = []
    education: List[str] = []
    languages: List[str] = []  # NEW FIELD
    source_pdf: Optional[str] = None
    download_timestamp: Optional[str] = None

class ScrapingRequest(BaseModel):
    search_query: str
    location: str = "Nigeria"
    max_profiles: int = 5
    
    class Config:
        json_schema_extra = {
            "example": {
                "search_query": "frontend developer",
                "location": "Nigeria",
                "max_profiles": 5
            }
        }

class ScrapingResponse(BaseModel):
    task_id: str
    status: str
    message: str
    search_query: str
    location: str
    max_profiles: int
    created_at: datetime

