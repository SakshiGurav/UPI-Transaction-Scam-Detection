# UPI-Transaction-Scam-Detection

#UPI SHIELD PRO

Overview

UPI SHIELD PRO is an AI-powered application designed to detect potential UPI payment scams by analyzing audio conversations. The app transcribes the audio, extracts key linguistic and contextual features, and applies machine learning techniques to assess the risk level of the conversation. The goal is to alert users about fraudulent calls before they fall victim to scams.

Project Setup
Prerequisites
Ensure you have the following installed:
Python 3.8+
pip (Python package manager)
Streamlit
Whisper (OpenAI’s speech recognition model)
Librosa (for audio processing)
NumPy, Pandas, Scikit-learn (for data analysis and ML modeling)
Pyttsx3 (for text-to-speech alerts)
TextBlob (for sentiment analysis)

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py

Dataset Used:
The application uses a simulated dataset for scam  conversation patterns:
Scam Conversations: Contains features like text length, sentiment polarity, keyword occurrences, and scam-related terminology.
Normal Conversations: Samples of everyday conversations for model training.
Live Audio Data: Users can upload or record calls for real-time analysis.
https://drive.google.com/drive/folders/1shIKKjHAB-Vcnp45HgHeXQYqv-p8Nt-6?usp=sharing

Tools & Technologies Implemented

Programming Language:
Python: Used for application logic, data processing, and machine learning.

Frameworks & Libraries:
Streamlit: For interactive web UI
OpenAI Whisper: For automatic speech recognition (ASR)
Librosa: For audio signal processing
Scikit-learn: For machine learning (Isolation Forest for anomaly detection)
TextBlob: For sentiment analysis
Pyttsx3: For text-to-speech feedback

Model Implementation:
Whisper Model (Speech-to-Text): Converts recorded or uploaded audio to text.
Feature Extraction: Text length, word count, sentiment polarity, keyword detection.
Anomaly Detection (Isolation Forest): Identifies fraudulent patterns in speech.
Keyword Matching: Flags critical scam words like "OTP," "bank block," "immediate payment."
Risk Assessment: Calculates scam probability based on anomaly scores and keyword density.

Execution Instructions:
Running the Application
Launch the app:
streamlit run app.py
Analyze a Call:
Record Live Audio (via browser)
Upload a Pre-recorded Audio File (.wav, .mp3)
View Results:
Scam probability score
Key detected scam words
Transcription of the call
Preventive security tips

Key Features:
 Real-time scam detection
 AI-based speech-to-text analysis
 Keyword-based scam identification
 Machine learning for fraud risk assessment
 User-friendly interface
 Simulated UPI transaction blocking (Demo Mode)
 Privacy-focused: No data storage

 Future Improvements:
Integration with live banking fraud databases
Real-time reporting to authorities
Multilingual support for Indian languages
Enhanced deep-learning-based fraud detection


