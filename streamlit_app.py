# =============================================================================
# FAKE JOB DETECTOR — STREAMLIT WEB APP
# File: streamlit_app.py
# Author: Divya Virkud
# Version: 2.0 | March 2026
# Description: Live fraud detection web app with red flag analysis
# =============================================================================
# RUN: streamlit run streamlit_app.py
# =============================================================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data (runs once)
try:
    nltk.data.find('vader_lexicon')
except:
    nltk.download('vader_lexicon', quiet=True)
try:
    nltk.data.find('stopwords')
except:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('wordnet')
except:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('punkt')
except:
    nltk.download('punkt', quiet=True)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Fake Job Detector | India Edition",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LOAD MODEL AND ARTIFACTS
# =============================================================================

@st.cache_resource
def load_artifacts():
    """Load all saved models and preprocessing objects."""
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('processed_data/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('processed_data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, vectorizer, scaler, feature_names

model, vectorizer, scaler, feature_names = load_artifacts()

# =============================================================================
# FRAUD KEYWORD LISTS (same as preprocessing)
# =============================================================================

INDIA_FRAUD_WORDS = [
    # Portal impersonation
    'linkedin selection', 'naukri selection', 'indeed hiring',
    'internshala selected', 'selected by linkedin', 'selected by naukri',
    # Gig/task scams
    'amazon work from home', 'flipkart hiring', 'zomato delivery partner',
    'swiggy job', 'meesho reseller', 'apna app', 'whatsapp job',
    # Crypto/payment
    'task based earning', 'telegram job', 'bitcoin payment',
    'cryptocurrency', 'gpay transfer', 'phonepe payment', 'paytm wallet',
    # Fake job types
    'data entry work from home', 'captcha typing', 'form filling job',
    'copy paste work', 'survey job', 'clicking job', 'like and share job'
]

URGENCY_WORDS = [
    'immediate', 'urgent', 'hurry', 'limited slots', 'today only',
    'expire soon', 'first come first serve', 'act now', 'don\'t miss',
    'limited time', 'closing soon', 'last chance', 'apply now'
]

OVERPROMISE_WORDS = [
    'guaranteed income', 'earn lakhs', 'no experience needed',
    'work 2 hours', 'easy money', 'get rich', '100% success',
    'assured salary', 'zero investment', 'lifetime income'
]

SUSPICIOUS_CONTACT = [
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
    'rediffmail.com', 'protonmail.com', 'tempmail.com'
]

# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def preprocess_text(text):
    """Clean and preprocess text (same as training pipeline)."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\+?\d[\d\s\-\(\)]{7,}\d', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return ' '.join(tokens)

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(job_title, company, location, description, 
                    requirements, benefits, employment_type,
                    required_experience, salary_range):
    """Extract all 42 engineered features from job posting."""
    
    # Combine all text
    full_text = f"{job_title} {company} {description} {requirements} {benefits}"
    
    # Clean text
    cleaned_desc = preprocess_text(description)
    cleaned_text = preprocess_text(full_text)
    
    # Initialize VADER
    sia = SentimentIntensityAnalyzer()
    
    features = {}
    
    # --- Text statistics ---
    features['desc_word_count']     = len(str(description).split())
    features['desc_char_count']     = len(str(description))
    features['desc_sentence_count'] = len(re.split(r'[.!?]', str(description)))
    features['title_word_count']    = len(str(job_title).split())
    words = str(description).split()
    features['avg_word_length'] = round(np.mean([len(w) for w in words]), 4) if words else 0
    
    # --- Punctuation abuse ---
    features['exclamation_count']  = full_text.count('!')
    features['question_mark_count'] = full_text.count('?')
    caps_count = sum(1 for c in full_text if c.isupper())
    features['caps_ratio'] = round(caps_count / max(len(full_text), 1), 4)
    
    # --- India fraud signals ---
    text_lower = full_text.lower()
    features['india_fraud_score'] = sum(1 for word in INDIA_FRAUD_WORDS if word in text_lower)
    features['urgency_score']     = sum(1 for word in URGENCY_WORDS if word in text_lower)
    features['overpromise_score'] = sum(1 for word in OVERPROMISE_WORDS if word in text_lower)
    
    # --- Contact red flags ---
    features['has_whatsapp_contact'] = int('whatsapp' in text_lower)
    features['has_suspicious_email'] = int(any(d in text_lower for d in SUSPICIOUS_CONTACT))
    features['has_phone_in_desc']    = int(bool(re.search(r'(\+91[\-\s]?)?[6-9]\d{9}', description)))
    
    # --- Fee and document flags ---
    fee_patterns = ['registration fee', 'processing fee', 'security deposit',
                    'training fee', 'pay to apply', 'refundable deposit']
    features['has_fee_language'] = int(any(p in text_lower for p in fee_patterns))
    
    doc_patterns = ['aadhar', 'aadhaar', 'pan card', 'passport copy',
                    'bank account', 'upi id', 'gpay', 'phonepay', 'paytm',
                    'selfie with id', 'video kyc']
    features['requests_personal_docs'] = int(any(p in text_lower for p in doc_patterns))
    
    # --- Portal fraud patterns ---
    linkedin_fraud = ['linkedin selection', 'selected by linkedin', 'linkedin hiring team']
    features['has_linkedin_fraud'] = int(any(p in text_lower for p in linkedin_fraud))
    
    portal_names = ['linkedin', 'naukri', 'indeed', 'internshala']
    selection_words = ['selection', 'selected', 'hiring', 'offer']
    has_portal = any(p in text_lower for p in portal_names)
    has_selection = any(s in text_lower for s in selection_words)
    features['has_portal_fraud_pattern'] = int(has_portal and has_selection)
    
    # --- Sentiment ---
    sentiment = sia.polarity_scores(str(description)[:512])
    features['sentiment_compound']  = round(sentiment['compound'], 4)
    features['sentiment_positive']  = round(sentiment['pos'], 4)
    features['sentiment_negative']  = round(sentiment['neg'], 4)
    features['sentiment_neutral']   = round(sentiment['neu'], 4)
    
    # --- Repetition ---
    words_list = str(description).lower().split()
    if len(words_list) >= 10:
        features['repetition_score'] = round(1 - len(set(words_list)) / len(words_list), 4)
    else:
        features['repetition_score'] = 0.0
    
    # --- Salary ---
    salary_str = str(salary_range)
    nums = re.findall(r'\d+', salary_str.replace(',', ''))
    if len(nums) >= 2:
        salary_min, salary_max = int(nums[0]), int(nums[1])
        salary_missing = 0
    elif len(nums) == 1:
        salary_min = salary_max = int(nums[0])
        salary_missing = 0
    else:
        salary_min = salary_max = 0
        salary_missing = 1
    
    features['salary_min']        = salary_min
    features['salary_max']        = salary_max
    features['salary_missing']    = salary_missing
    features['salary_range_width'] = salary_max - salary_min
    
    is_fresher = any(word in str(required_experience).lower() 
                     for word in ['no experience', 'fresher', 'entry', '0 year'])
    features['salary_too_high_for_fresher'] = int(salary_max > 250000 and is_fresher)
    features['salary_unrealistic'] = int(salary_max > 800000)
    
    # --- Company info ---
    features['company_profile_missing'] = int(str(company).strip() == '' or str(company).lower() == 'nan')
    features['company_profile_length']  = len(str(company))
    features['has_company_logo']        = 0  # Cannot determine from text
    features['has_questions']           = 0  # Cannot determine from text
    
    # --- Job title ---
    vague_patterns = ['data entry', 'home based', 'work from home', 'part time',
                     'earn money', 'online job', 'simple job', 'easy job', 
                     'typing work', 'form filling', 'copy paste']
    features['vague_job_title'] = int(any(p in str(job_title).lower() for p in vague_patterns))
    
    # --- Requirements and benefits ---
    features['requirements_missing'] = int(str(requirements).strip() == '' or str(requirements).lower() == 'nan')
    features['requirements_length']  = len(str(requirements))
    features['benefits_missing']     = int(str(benefits).strip() == '' or str(benefits).lower() == 'nan')
    
    # --- Employment type encoding ---
    emp_map = {'full-time': 0, 'part-time': 1, 'contract': 2, 'temporary': 3, 'other': 4}
    features['employment_type_encoded'] = emp_map.get(str(employment_type).lower(), 5)
    
    # --- Experience encoding ---
    exp_str = str(required_experience).lower()
    if 'not applicable' in exp_str or exp_str.strip() == '':
        features['experience_encoded'] = -1
    elif 'entry' in exp_str or 'internship' in exp_str:
        features['experience_encoded'] = 0
    elif 'associate' in exp_str or '1-2' in exp_str:
        features['experience_encoded'] = 1
    elif 'mid-senior' in exp_str or '2-5' in exp_str:
        features['experience_encoded'] = 2
    else:
        features['experience_encoded'] = 0
    
    # --- Location ---
    generic_loc = ['anywhere', 'worldwide', 'remote', 'work from home',
                   'india', 'pan india', 'all india', 'multiple cities']
    features['generic_location'] = int(any(g in str(location).lower() for g in generic_loc))
    
    # --- Missing fields score ---
    features['missing_fields_score'] = (features['company_profile_missing'] +
                                        features['requirements_missing'] +
                                        features['benefits_missing'] +
                                        features['salary_missing'])
    
    # --- Telecommuting ---
    features['telecommuting'] = int('remote' in text_lower or 'work from home' in text_lower)
    
    return features, cleaned_text

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def predict_fraud(job_title, company, location, description,
                 requirements, benefits, employment_type,
                 required_experience, salary_range):
    """Make fraud prediction and return probability + red flags."""
    
    # Extract features
    features_dict, cleaned_text = extract_features(
        job_title, company, location, description,
        requirements, benefits, employment_type,
        required_experience, salary_range
    )
    
    # TF-IDF vectorization
    tfidf_vec = vectorizer.transform([cleaned_text])
    
    # Engineered features
    feature_order = [
        'desc_word_count', 'desc_char_count', 'desc_sentence_count',
        'title_word_count', 'avg_word_length',
        'exclamation_count', 'question_mark_count', 'caps_ratio',
        'india_fraud_score', 'urgency_score', 'overpromise_score',
        'has_whatsapp_contact', 'has_suspicious_email', 'has_phone_in_desc',
        'has_fee_language', 'requests_personal_docs',
        'has_linkedin_fraud', 'has_portal_fraud_pattern',
        'sentiment_compound', 'sentiment_positive',
        'sentiment_negative', 'sentiment_neutral',
        'repetition_score',
        'salary_min', 'salary_max', 'salary_missing', 'salary_range_width',
        'salary_too_high_for_fresher', 'salary_unrealistic',
        'company_profile_missing', 'company_profile_length',
        'has_company_logo', 'has_questions',
        'vague_job_title',
        'requirements_missing', 'requirements_length', 'benefits_missing',
        'employment_type_encoded', 'experience_encoded',
        'generic_location', 'missing_fields_score', 'telecommuting'
    ]
    
    engineered_arr = np.array([features_dict[f] for f in feature_order]).reshape(1, -1)
    engineered_scaled = scaler.transform(engineered_arr)
    
    # Combine features
    import scipy.sparse as sp
    X_combined = sp.hstack([tfidf_vec, sp.csr_matrix(engineered_scaled)])
    
    # Predict
    fraud_prob = model.predict_proba(X_combined.toarray())[0][1]
    prediction = int(fraud_prob >= 0.5)
    
    # Identify red flags
    red_flags = []
    if features_dict['india_fraud_score'] > 0:
        red_flags.append(f"🚩 Contains {features_dict['india_fraud_score']} India fraud keywords")
    if features_dict['urgency_score'] > 1:
        red_flags.append(f"🚩 Uses {features_dict['urgency_score']} urgency words")
    if features_dict['has_whatsapp_contact']:
        red_flags.append("🚩 Mentions WhatsApp contact")
    if features_dict['has_suspicious_email']:
        red_flags.append("🚩 Uses free email domain (Gmail/Yahoo/Outlook)")
    if features_dict['has_fee_language']:
        red_flags.append("🚩 Mentions registration/processing fees")
    if features_dict['requests_personal_docs']:
        red_flags.append("🚩 Requests sensitive documents (Aadhar/PAN/UPI)")
    if features_dict['has_portal_fraud_pattern']:
        red_flags.append("🚩 Impersonates job portal (LinkedIn/Naukri/Indeed)")
    if features_dict['salary_too_high_for_fresher']:
        red_flags.append("🚩 Unrealistic salary for fresher role")
    if features_dict['company_profile_missing']:
        red_flags.append("🚩 Missing company information")
    if features_dict['vague_job_title']:
        red_flags.append("🚩 Vague job title (data entry/form filling)")
    if features_dict['missing_fields_score'] >= 3:
        red_flags.append("🚩 Multiple missing fields (suspicious)")
    if features_dict['exclamation_count'] > 3:
        red_flags.append(f"🚩 Excessive punctuation ({features_dict['exclamation_count']} exclamation marks)")
    
    return prediction, fraud_prob, red_flags, features_dict

# =============================================================================
# STREAMLIT UI
# =============================================================================

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fraud-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .safe-job {
        background-color: #d4edda;
        color: #155724;
    }
    .fraud-job {
        background-color: #f8d7da;
        color: #721c24;
    }
    .red-flag {
        background-color: #fff3cd;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-left: 4px solid #ff9800;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔍 Fake Job Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">India Edition | Powered by XGBoost + NLP</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 About This App")
    st.write("""
    This ML-powered tool detects fraudulent job postings targeting Indian job seekers.
    
    **Features:**
    - 42 engineered fraud signals
    - Portal impersonation detection
    - India-specific scam patterns
    - Real-time fraud scoring
    
    **Built by:** Divya Virkud  
    **Model:** XGBoost (F1: 0.93+)  
    **Version:** 2.0 | March 2026
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Common Red Flags")
    st.write("""
    - WhatsApp/Telegram contact
    - Requests for Aadhar/PAN/UPI
    - Registration/processing fees
    - Unrealistic salaries
    - Portal impersonation
    - Vague job titles
    """)

# Initialize session state for form clearing
if 'clear_form' not in st.session_state:
    st.session_state.clear_form = False

# Clear form if button was clicked
if st.session_state.clear_form:
    st.session_state.clear_form = False
    st.rerun()

# Main form
st.markdown("### 📝 Enter Job Posting Details")

# Use session state keys to control form values
col1, col2 = st.columns(2)

with col1:
    job_title = st.text_input("Job Title *", 
                              placeholder="e.g., Data Analyst",
                              key="job_title")
    company = st.text_input("Company Name", 
                            placeholder="e.g., TCS",
                            key="company")
    location = st.text_input("Location", 
                             placeholder="e.g., Mumbai, Maharashtra",
                             key="location")
    employment_type = st.selectbox("Employment Type", 
                                   ["Full-time", "Part-time", "Contract", "Temporary", "Other"],
                                   key="employment_type")

with col2:
    required_experience = st.text_input("Required Experience", 
                                        placeholder="e.g., 0-2 years / Fresher",
                                        key="experience")
    salary_range = st.text_input("Salary Range", 
                                  placeholder="e.g., ₹3-5 LPA or 30000-50000",
                                  key="salary")
    requirements = st.text_area("Requirements", 
                                placeholder="Educational qualifications, skills needed...",
                                height=100,
                                key="requirements")

description = st.text_area("Job Description *", 
                           placeholder="Detailed job description, responsibilities, etc.",
                           height=200,
                           key="description")

benefits = st.text_area("Benefits", 
                        placeholder="Perks, benefits offered...",
                        height=100,
                        key="benefits")

# Predict button with Clear Form option
col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    analyze_btn = st.button("🔍 Analyze Job Posting", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🔄 Clear Form", use_container_width=True):
        # Clear all form keys from session state
        keys_to_clear = ['job_title', 'company', 'location', 'employment_type',
                        'experience', 'salary', 'requirements', 'description', 'benefits']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.clear_form = True
        st.rerun()

if analyze_btn:
    
    if not job_title or not description:
        st.error("⚠️ Please fill in at least Job Title and Description")
    else:
        with st.spinner("Analyzing job posting..."):
            
            prediction, fraud_prob, red_flags, features = predict_fraud(
                job_title, company, location, description,
                requirements, benefits, employment_type,
                required_experience, salary_range
            )
            
            # Display result
            st.markdown("---")
            st.markdown("## 🎯 Analysis Result")
            
            fraud_percentage = round(fraud_prob * 100, 1)
            
            if prediction == 1:
                st.markdown(f'<div class="fraud-score fraud-job">⚠️ FRAUD DETECTED<br>{fraud_percentage}% Fraud Probability</div>', 
                           unsafe_allow_html=True)
                st.error("🚨 This job posting shows strong indicators of fraud. Proceed with extreme caution!")
            else:
                st.markdown(f'<div class="fraud-score safe-job">✅ LIKELY LEGITIMATE<br>{fraud_percentage}% Fraud Probability</div>', 
                           unsafe_allow_html=True)
                st.success("✓ This job posting appears legitimate. Always verify independently!")
            
            # Red flags section
            if red_flags:
                st.markdown("### 🚩 Red Flags Detected")
                for flag in red_flags:
                    st.markdown(f'<div class="red-flag">{flag}</div>', unsafe_allow_html=True)
            else:
                st.info("✓ No major red flags detected")
            
            # Feature breakdown
            with st.expander("📊 Feature Analysis Breakdown"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("India Fraud Keywords", features['india_fraud_score'])
                    st.metric("Urgency Words", features['urgency_score'])
                    st.metric("Description Length", f"{features['desc_word_count']} words")
                
                with col2:
                    st.metric("Overpromise Score", features['overpromise_score'])
                    st.metric("Missing Fields", features['missing_fields_score'])
                    st.metric("Sentiment", round(features['sentiment_compound'], 2))
                
                with col3:
                    st.metric("Portal Fraud Pattern", "Yes" if features['has_portal_fraud_pattern'] else "No")
                    st.metric("Requests Documents", "Yes" if features['requests_personal_docs'] else "No")
                    st.metric("Fee Language", "Yes" if features['has_fee_language'] else "No")
            
            # Safety tips
            st.markdown("---")
            st.markdown("### 💡 Safety Tips")
            st.info("""
            ✓ Research company on official website  
            ✓ Never pay registration/processing fees  
            ✓ Don't share Aadhar/PAN before offer letter  
            ✓ Verify job via company's official channels  
            ✓ Be wary of WhatsApp/Telegram contacts  
            ✓ Check salary expectations on Glassdoor/AmbitionBox
            """)
            
            # Analyze another job button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Analyze Another Job Posting", 
                           type="secondary", 
                           use_container_width=True):
                    # Clear all form keys from session state
                    keys_to_clear = ['job_title', 'company', 'location', 'employment_type',
                                    'experience', 'salary', 'requirements', 'description', 'benefits']
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.session_state.clear_form = True
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>Built with Streamlit + XGBoost | Trained on 17,880+ job postings</p>
    <p>For educational purposes only. Always verify job postings independently.</p>
</div>
""", unsafe_allow_html=True)
