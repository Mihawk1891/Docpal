import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import speech_recognition as sr
from textblob import TextBlob
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
import spacy
import plotly.graph_objs as go

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

class EnhancedAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.vectorizer = CountVectorizer()
        self.classifier = self.load_or_train_classifier()

    def load_or_train_classifier(self):
        if os.path.exists('psychological_model.pkl'):
            with open('psychological_model.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            # Expanded training data
            X = [
                "I feel happy and excited about life",
                "I'm angry and frustrated with everything",
                "I'm confused and don't know what to do",
                "I'm excited and looking forward to the future",
                "I feel sad and hopeless",
                "I'm anxious and worried about everything",
                "I feel calm and at peace",
                "I'm bored and uninterested in things"
            ]
            y = ["joy", "anger", "confusion", "excitement", "sadness", "anxiety", "calmness", "boredom"]
            X_vectorized = self.vectorizer.fit_transform(X)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_vectorized, y)
            with open('psychological_model.pkl', 'wb') as f:
                pickle.dump(clf, f)
            return clf

    def analyze_text(self, text):
        sentiment = self.sia.polarity_scores(text)
        blob = TextBlob(text)
        subjectivity = blob.sentiment.subjectivity
        psychological_state = self.classify_psychological_state(text)
        entities = self.extract_entities(text)
        return sentiment, subjectivity, psychological_state, entities

    def classify_psychological_state(self, text):
        vectorized_text = self.vectorizer.transform([text])
        return self.classifier.predict(vectorized_text)[0]

    def extract_entities(self, text):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def analyze_audio(self, audio_file):
        try:
            y, sr = librosa.load(audio_file)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            pitch = np.mean(librosa.feature.pitch(y=y, sr=sr))
            return {"tempo": tempo, "pitch": pitch}
        except Exception as e:
            st.error(f"Error analyzing audio: {str(e)}")
            return None

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand the audio"
    except sr.RequestError:
        return "Could not request results from the speech recognition service"
    except Exception as e:
        return f"An error occurred during transcription: {str(e)}"

def generate_insights(sentiment, subjectivity, psychological_state, entities, audio_features=None):
    insights = []

    sentiment_mapping = {
        (-1.0, -0.6): "very negative",
        (-0.6, -0.2): "somewhat negative",
        (-0.2, 0.2): "neutral",
        (0.2, 0.6): "somewhat positive",
        (0.6, 1.0): "very positive"
    }

    for range_, description in sentiment_mapping.items():
        if range_[0] <= sentiment['compound'] < range_[1]:
            insights.append(f"The speaker expresses a {description} sentiment.")
            break

    insights.append(f"The speech contains {'more subjective' if subjectivity > 0.5 else 'more objective'} language.")
    insights.append(f"The speaker appears to be in a state of {psychological_state}.")

    if entities:
        insights.append("Identified entities in the text:")
        for entity, label in entities:
            insights.append(f"- {entity} ({label})")

    if audio_features:
        if audio_features['tempo'] > 120:
            insights.append("The speaker's pace is quick, suggesting urgency or excitement.")
        elif audio_features['tempo'] < 80:
            insights.append("The speaker's pace is slow, suggesting calmness or thoughtfulness.")

        if audio_features['pitch'] > 200:
            insights.append("The speaker's higher pitch may indicate stress or enthusiasm.")
        elif audio_features['pitch'] < 150:
            insights.append("The speaker's lower pitch may indicate confidence or calmness.")

    return insights

def plot_sentiment(sentiment):
    labels = ['Negative', 'Neutral', 'Positive']
    values = [sentiment['neg'], sentiment['neu'], sentiment['pos']]

    fig = go.Figure(data=[go.Bar(x=labels, y=values)])
    fig.update_layout(title='Sentiment Analysis', yaxis_title='Score')
    return fig

def main():
    st.title("Enhanced Sentiment and Psychological Insight Analyzer")

    analyzer = EnhancedAnalyzer()

    input_type = st.radio("Select input type:", ("Text", "Audio"))

    if input_type == "Text":
        analyze_text_input(analyzer)
    elif input_type == "Audio":
        analyze_audio_input(analyzer)

    st.sidebar.markdown("### About")
    st.sidebar.info("This app provides advanced sentiment analysis and psychological insights from text or audio input.")

def analyze_text_input(analyzer):
    user_input = st.text_area("Enter your text here:")
    if st.button("Analyze Text"):
        if user_input:
            sentiment, subjectivity, psychological_state, entities = analyzer.analyze_text(user_input)
            insights = generate_insights(sentiment, subjectivity, psychological_state, entities)
            display_analysis_results(sentiment, subjectivity, psychological_state, entities, insights)
        else:
            st.warning("Please enter some text to analyze.")

def analyze_audio_input(analyzer):
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if st.button("Analyze Audio"):
        if audio_file:
            text = transcribe_audio(audio_file)
            st.subheader("Transcribed Text:")
            st.write(text)

            sentiment, subjectivity, psychological_state, entities = analyzer.analyze_text(text)
            audio_features = analyzer.analyze_audio(audio_file)
            insights = generate_insights(sentiment, subjectivity, psychological_state, entities, audio_features)
            display_analysis_results(sentiment, subjectivity, psychological_state, entities, insights, audio_features)
        else:
            st.warning("Please upload an audio file to analyze.")

def display_analysis_results(sentiment, subjectivity, psychological_state, entities, insights, audio_features=None):
    st.subheader("Analysis Results:")
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Subjectivity: {subjectivity:.2f}")
    st.write(f"Psychological State: {psychological_state}")
    
    if entities:
        st.write("Entities:")
        for entity, label in entities:
            st.write(f"- {entity} ({label})")
    
    if audio_features:
        st.write(f"Audio Features: {audio_features}")

    st.plotly_chart(plot_sentiment(sentiment))

    st.subheader("Psychological Insights:")
    for insight in insights:
        st.write(f"- {insight}")

if __name__ == "__main__":
    main()