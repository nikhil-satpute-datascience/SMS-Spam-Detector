import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sms_spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
/* App background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
}

/* Hide Streamlit default items */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Main content spacing */
.block-container {
    padding-top: 80px;
    max-width: 900px;
}

/* Title */
.title {
    text-align: center;
    font-size: 54px;
    font-weight: 900;
    color: #f8fafc;
    margin-bottom: 14px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #cbd5e1;
    line-height: 1.6;
    margin-bottom: 45px;
}

/* Input label */
label {
    color: #f8fafc !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

/* Text area */
textarea {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 18px !important;
    border: 2px solid #475569 !important;
    font-size: 18px !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    color: white;
    border: none;
    border-radius: 16px;
    padding: 15px 22px;
    font-size: 20px;
    font-weight: 800;
    width: 100%;
    margin-top: 20px;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #4338ca);
    color: white;
    transform: scale(1.01);
}

/* Result boxes */
.spam-result {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    color: #7f1d1d;
    padding: 26px;
    border-radius: 18px;
    text-align: center;
    font-size: 30px;
    font-weight: 900;
    border: 2px solid #ef4444;
    margin-top: 30px;
}

.safe-result {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    color: #14532d;
    padding: 26px;
    border-radius: 18px;
    text-align: center;
    font-size: 30px;
    font-weight: 900;
    border: 2px solid #22c55e;
    margin-top: 30px;
}

/* Description */
.result-desc {
    background: rgba(255, 255, 255, 0.10);
    color: #e2e8f0;
    padding: 18px;
    border-radius: 14px;
    margin-top: 15px;
    font-size: 17px;
    line-height: 1.6;
}

/* About section */
.about-box {
    background: rgba(255, 255, 255, 0.10);
    padding: 28px;
    border-radius: 20px;
    margin-top: 45px;
    border: 1px solid rgba(255,255,255,0.15);
}

.about-title {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 12px;
}

.about-text {
    color: #cbd5e1;
    font-size: 17px;
    line-height: 1.7;
}

.tech-pill {
    display: inline-block;
    background: rgba(99, 102, 241, 0.20);
    color: #c7d2fe;
    padding: 8px 14px;
    border-radius: 999px;
    margin: 5px;
    font-size: 14px;
    font-weight: 700;
}

.custom-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)

# Prediction function
def predict_sms(message):
    message_tfidf = tfidf.transform([message])
    prediction = model.predict(message_tfidf)[0]

    if prediction == 1:
        return "Spam"
    else:
        return "Legitimate"

# Main UI
st.markdown(
    '<div class="title">📩 SMS Spam Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '''
    <div class="subtitle">
    Paste any SMS message below and the AI model will classify it as 
    <b>Spam</b> or <b>Legitimate</b>.
    </div>
    ''',
    unsafe_allow_html=True
)

message = st.text_area(
    "Enter SMS Message",
    placeholder="Type or paste your SMS message here...",
    height=180
)

check = st.button("Check Message")

if check:
    if message.strip() == "":
        st.warning("Please enter an SMS message first.")
    else:
        result = predict_sms(message)

        if result == "Spam":
            st.markdown(
                '<div class="spam-result">🚨 Spam Message Detected</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '''
                <div class="result-desc">
                This message looks suspicious. Avoid clicking unknown links, sharing personal details, 
                or trusting prize and urgent payment messages.
                </div>
                ''',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="safe-result">✅ Legitimate Message</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '''
                <div class="result-desc">
                This message looks normal based on the trained machine learning model.
                </div>
                ''',
                unsafe_allow_html=True
            )

# About section
st.markdown(
    '''
    <div class="about-box">
        <div class="about-title">About This Project</div>
        <div class="about-text">
            This web application classifies SMS messages as Spam or Legitimate using Natural Language Processing 
            and Machine Learning. The text is converted into numerical features using TF-IDF, and the final 
            prediction is made using a trained Support Vector Machine model.
        </div>
        <br>
        <span class="tech-pill">Python</span>
        <span class="tech-pill">Streamlit</span>
        <span class="tech-pill">Scikit-learn</span>
        <span class="tech-pill">TF-IDF</span>
        <span class="tech-pill">SVM</span>
        <span class="tech-pill">NLP</span>
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="custom-footer">Made by Nikhil Satpute | SMS Spam Detection ML Project</div>',
    unsafe_allow_html=True
)