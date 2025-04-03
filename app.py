import os
import streamlit as st
import whisper
import numpy as np
import pandas as pd
import pyttsx3
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import librosa
import io

audio_bytes = None  

# Setting page config with dark background
st.set_page_config(
    page_title="UPI SHIELD PRO",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme with better contrast
st.markdown("""
<style>
    body {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .header {
        color: #00D1B2;
        text-align: center;
        font-size: 5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 3px #000000;
    }
    .subheader {
        color: #AAAAAA;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .scam-alert {
        border-left: 5px solid #FF3860;
        padding: 1.5rem;
        background-color: #1A1D23;
        border-radius: 8px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .safe-call {
        border-left: 5px solid #00D1B2;
        padding: 1.5rem;
        background-color: #1A1D23;
        border-radius: 8px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .metric-box {
        background-color: #1A1D23;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        border: 1px solid #303030;
    }
    .upload-box {
        border: 2px dashed #00D1B2;
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        margin-bottom: 2rem;
        background-color: rgba(0, 209, 178, 0.05);
    }
    .stTextArea>div>div>textarea {
        background-color: #1A1D23 !important;
        color: #FFFFFF !important;
    }
    .footer {
        color: #666666;
        text-align: center;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
    .highlight {
        color: #00D1B2;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="header">UPI SHIELD PRO</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">AI-powered protection against UPI payment scams</p>', unsafe_allow_html=True)

# Upload/Record Section

with st.container():
    st.markdown("""
    <div class="upload-box">
        <h2 style="color: #FFFFFF;">Analyze Call</h2>
        <p style="color: #AAAAAA;">Record live audio (Chrome/Firefox) or upload</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎙️ Record Live", "📁 Upload Recording"])
    
    with tab1:
        try:
            audio_bytes = st.audio(
                start_recording=True,
                sample_rate=16000,
                key="recorder",
                format="wav"
            )
        except TypeError:
            st.warning("Live recording requires Streamlit 1.29.0+")
            if st.button("🎤 Simulate Recording (Demo)"):
                audio_bytes = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
                st.audio(audio_bytes, format="audio/wav")
    
    with tab2:
        uploaded_file = st.file_uploader("", type=["wav", "mp3"], label_visibility="collapsed")

# Loading Models (cached)

@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("small")
    scam_data = pd.DataFrame({
        "Text_Length": np.random.randint(300, 600, 50),
        "Word_Count": np.random.randint(80, 150, 50),
        "Sentiment": np.random.uniform(-0.9, -0.3, 50)
    })
    normal_data = pd.DataFrame({
        "Text_Length": np.random.randint(50, 200, 50),
        "Word_Count": np.random.randint(10, 40, 50),
        "Sentiment": np.random.uniform(-0.2, 0.5, 50)
    })
    full_data = pd.concat([scam_data, normal_data])
    scaler = StandardScaler()
    X = scaler.fit_transform(full_data)
    model = IsolationForest(contamination=0.4, random_state=42)
    model.fit(X)
    return whisper_model, model, scaler

whisper_model, anomaly_model, scaler = load_models()

# Processing Functions

def process_audio(uploaded_file):
    audio_bytes = uploaded_file.read()
    with io.BytesIO(audio_bytes) as audio_file:
        y, _ = librosa.load(audio_file, sr=16000, mono=True)
    return y.astype(np.float32)

SCAM_KEYWORDS = {
    # Payment/Account Threats (25 terms)
    "block": 3, "suspend": 3, "freeze": 3, "deactivate": 3, "terminate": 3,
    "restrict": 2, "limit": 2, "disabled": 2, "locked": 2, "closed": 2,
    "unauthorized": 3, "hacked": 3, "compromised": 3, "breached": 3,
    "fraudulent": 3, "illegal": 2, "action": 2, "penalty": 2, "fine": 2,
    "case filed": 3, "police complaint": 3, "FIR": 3, "legal action": 3,
    "court order": 3, "judgment": 2,

    # Verification Scams (30 terms)
    "OTP": 4, "one time password": 4, "verification code": 4, 
    "authenticate": 3, "validate": 3, "confirm": 3, "verify": 4,
    "KYC": 4, "know your customer": 4, "update KYC": 4, "complete KYC": 4,
    "link aadhaar": 4, "aadhaar card": 3, "pan card": 3, "PAN number": 3,
    "biometric": 3, "fingerprint": 2, "iris scan": 2, "e-sign": 2,
    "digital signature": 2, "VID": 2, "virtual ID": 2, "UIDAI": 2,
    "bank details": 3, "account number": 3, "IFSC": 3, "CVV": 4,
    "debit card": 3, "credit card": 3, "expiry date": 2, "UPI PIN": 4,

    # Fake Rewards/Lotteries (25 terms)
    "reward": 4, "prize": 4, "winner": 4, "won": 4, "lucky": 4,
    "draw": 4, "contest": 4, "lottery": 4, "raffle": 3, "voucher": 3,
    "gift": 3, "giveaway": 3, "bonus": 3, "coupon": 2, "offer": 3,
    "Amazon": 3, "Flipkart": 3, "Myntra": 2, "Paytm": 4, "PhonePe": 4,
    "Google Pay": 4, "BHIM": 4, "free": 3, "cashback": 4, "discount": 3,

    # Urgency Triggers (20 terms)
    "urgent": 4, "immediately": 4, "now": 3, "right now": 4,
    "last chance": 4, "final warning": 4, "limited time": 3,
    "today only": 3, "within 24 hours": 3, "expiring": 3,
    "deadline": 3, "immediate action": 4, "quick": 3,
    "hurry": 3, "time sensitive": 3, "important": 3,
    "critical": 3, "emergency": 3, "asap": 3, "attention": 3,

    # Payment Requests (25 terms)
    "send money": 4, "transfer": 3, "pay": 3, "payment": 3,
    "processing fee": 4, "charges": 3, "tax": 3, "deposit": 4,
    "security amount": 4, "advance": 3, "token": 3, "registration": 3,
    "subscription": 3, "membership": 2, "activation": 3,
    "unlock": 3, "clear": 3, "settle": 3, "balance": 3,
    "outstanding": 3, "due": 3, "pending": 3, "refund": 3,
    "return": 2, "reversal": 2,

    # Social Engineering (25 terms)
    "customer care": 3, "support": 2, "helpline": 2,
    "technical team": 3, "security": 3, "fraud": 3,
    "cyber crime": 3, "nodal officer": 2, "manager": 2,
    "executive": 2, "representative": 2, "official": 2,
    "agent": 2, "authority": 2, "department": 2,
    "government": 3, "RBI": 3, "TRAI": 2, "TDS": 2,
    "income tax": 3, "GST": 2, "ministry": 2,
    "police": 3, "cyber cell": 3, "CBI": 3
}

def extract_features(text):
    text = text.lower()
    features = {
        "length": len(text),
        "words": len(text.split()),
        "sentiment": TextBlob(text).sentiment.polarity,
        "scam_score": 0,
        "critical_words": 0,
        "urgent_phrases": 0
    }
    for keyword, weight in SCAM_KEYWORDS.items():
        count = text.count(keyword.lower())
        if count > 0:
            features["scam_score"] += weight * count
            if weight >= 3:
                features["critical_words"] += count
            if "urgent" in keyword or "immediate" in keyword:
                features["urgent_phrases"] += count
    features["exclamation"] = text.count("!") / max(1, features["words"])
    features["question"] = text.count("?") / max(1, features["words"])
    features["speech_rate"] = features["words"] / max(1, len(text.split())/10)
    return features

# Results Displaying with Dark Theme

if uploaded_file or (audio_bytes is not None):
    if audio_bytes and not uploaded_file:  
        st.audio(audio_bytes, format="audio/wav")
        try:
            
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except:
            st.error("Error processing recording. Please try again.")
            st.stop()
    else:  
        st.audio(uploaded_file)
        audio_array = process_audio(uploaded_file)
    
    with st.spinner("🔍 Analyzing call..."):
        try:
            result = whisper_model.transcribe(audio_array)
            transcript = result["text"]
            
            # Extract features
            features = extract_features(transcript)
            features_scaled = scaler.transform([[features["length"], features["words"], features["sentiment"]]])
            
            # Calculate probability
            anomaly_score = (1 - anomaly_model.decision_function(features_scaled)[0]) / 2
            keyword_score = min(features["scam_score"] / 50, 1.0)
            scam_prob = min(0.6 * anomaly_score + 0.4 * keyword_score, 1.0)
            
            if scam_prob > 0.45:
               try:
                  st.toast("🚨 SCAM ALERT: UPI payment request detected!", icon="⚠️")
               except:
                  st.markdown("""
                  <script>alert("🚨 SCAM ALERT!");</script>
                  """, unsafe_allow_html=True)

            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric-box">
                    <h3 style="color: #AAAAAA;">SCAM RISK</h3>
                    <h1 style="color: #FF3860; margin: 0;">{:.0f}%</h1>
                </div>
                """.format(scam_prob*100), unsafe_allow_html=True)
            
            # Results
            if scam_prob > 0.45 or features["critical_words"] >= 2:
                st.markdown("""
                <div class="scam-alert">
                    <h2>🚨 Dangerous Call Detected</h2>
                    <p>This call matches known UPI scam patterns. <strong>Do not share any information!</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Showing top scam phrases found
                found_keywords = sorted(
                    [kw for kw in SCAM_KEYWORDS if kw in transcript.lower()],
                    key=lambda x: -SCAM_KEYWORDS[x]
                )[:10]
                
                st.warning(f"**Scam indicators found**: {', '.join(found_keywords)}")
                
                try:
                   st.toast("🚨 SCAM ALERT: UPI payment request detected!", icon="⚠️")
                except:
                   st.markdown("""
                   <script>alert("🚨 SCAM ALERT!");</script>
                   """, unsafe_allow_html=True)
    
                # Feedback button
                st.warning("False positive? Let us know!")
                if st.button("👎 This is NOT a scam"):
                   st.session_state.feedback = "false_positive"
                   st.info("Thanks for your feedback! Our team will review this call.")
    
                # Simulated cancellation
                st.error("🚨 UPI Transaction Blocked (Simulated)")
                with st.expander("⚡ Cancel Transaction (Demo)"):
                    st.warning("In a real app, this would trigger a UPI cancellation request.")
                    if st.button("🗑️ Cancel Last UPI Payment (Simulated)"):
                       st.success("✅ Simulated: Transaction cancelled! (This is a demo)")
                       st.balloons()

                # Prevention tips - Dark Theme Version
                st.markdown("""
                <div style="background-color: #1e293b; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #00D1B2; margin: 1.5rem 0;">
                    <h3 style="color: #00D1B2; margin-top: 0;">🛡️ Protect Yourself</h3>
                    <ul style="color: #e2e8f0; padding-left: 1.5rem;">
                    <li style="margin-bottom: 0.5rem;">Never share OTPs or UPI PIN</li>
                    <li style="margin-bottom: 0.5rem;">Don't send any payments to unknown callers</li>
                    <li style="margin-bottom: 0.5rem;">Block the caller immediately</li>
                    <li style="margin-bottom: 0.5rem;">Report to <a href="https://cybercrime.gov.in" target="_blank" style="color: #38bdf8; text-decoration: none;">cybercrime.gov.in</a></li>
                    <li>Contact your bank's official helpline to verify</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                
            # Transcript section
            st.subheader("Call Transcript")
            st.text_area("", transcript, height=200, label_visibility="collapsed")
            
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")

# 6. Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #576574; font-size: 0.9rem;">
    <p>🔒 Your audio is processed locally and never stored</p>
    <p>For emergencies, call your bank immediately</p>
</div>
""", unsafe_allow_html=True)