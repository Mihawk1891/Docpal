# SameyAI

#Problem_1 (Only Explorer Mode)
## Overview

This project creates an interactive web application that combines natural language processing (NLP) and machine learning techniques to provide enhanced sentiment analysis and psychological insights from both text and audio inputs. The application leverages various NLP libraries and machine learning algorithms to offer a comprehensive analysis of emotional states and psychological conditions based on user-provided text or audio data.

## Key Features

- Sentiment Analysis: Determines the overall sentiment of text input using advanced NLP techniques.
- Psychological State Classification: Identifies psychological states such as joy, anger, confusion, excitement, sadness, anxiety, calmness, and boredom.
- Entity Extraction: Detects named entities in the input text.
- Audio Processing: Transcribes audio files into text and analyzes tempo and pitch characteristics.
- Interactive Visualization: Displays sentiment analysis results in a bar chart.
- User-Friendly Interface: Utilizes Streamlit for easy interaction and visualization.

## Technologies Used

- Streamlit: For creating the interactive web interface
- NLTK: Natural Language Toolkit for text processing and sentiment analysis
- spaCy: Modern natural language understanding library
- scikit-learn: Machine learning algorithms for classification
- TextBlob: Simple API for diving into common NLP tasks
- Librosa: Audio analysis library
- Speech Recognition: For transcribing audio files

## Project Structure

The project consists of several key components:

1. EnhancedAnalyzer Class: Centralizes all analysis functionality
2. Text Analysis Module: Handles sentiment scoring, subjectivity detection, and psychological state classification
3. Audio Processing Module: Transcribes audio and extracts tempo and pitch features
4. Visualization Module: Creates interactive charts using Plotly
5. Main Application: Manages user input selection and results display

## Implementation Highlights

- Custom Psychological State Classifier: Trains a machine learning model to identify specific emotional states
- Advanced Sentiment Scoring: Uses compound scores for nuanced sentiment analysis
- Entity Extraction: Leverages spaCy's named entity recognition capabilities
- Audio Feature Extraction: Analyzes tempo and pitch characteristics of speech

## Potential Applications

- Mental Health Support Systems
- Customer Service Analytics
- Social Media Monitoring Tools
- Content Creation Assistance

## Future Improvements

- Integration with additional NLP libraries for enhanced feature extraction
- Development of more sophisticated machine learning models for improved accuracy
- Expansion to support multiple languages
- Incorporation of real-time feedback mechanisms

## Getting Started

To run the application locally:

1. Install required dependencies using pip:
   ```
   pip install streamlit nltk spacy scikit-learn textblob librosa speech_recognition plotly
   ```

2. Download necessary NLTK data:
   ```
   python -m nltk.downloader vader_lexicon punkt averaged_perceptron_tagger maxent_ne_chunker words
   ```

3. Load the spaCy English model:
   ```
   python -m spacy download en_core_web_sm
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

