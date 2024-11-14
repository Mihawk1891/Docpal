
## • Problem_1
This project creates an interactive web application combining natural language processing (NLP) and machine learning techniques to enhance sentiment analysis and psychological insights from text and audio inputs. The application leverages various NLP libraries and machine learning algorithms to comprehensively analyze emotional states and psychological conditions based on user-provided text or audio data.

### Key Features of this solution

- Sentiment Analysis: Determines the overall sentiment of text input using advanced NLP techniques.
- Psychological State Classification: Identifies psychological states such as joy, anger, confusion, excitement, sadness, anxiety, calmness, and boredom.
- Entity Extraction: Detects named entities in the input text.
- Audio Processing: Transcribes audio files into text and analyzes tempo and pitch characteristics.
- Interactive Visualization: Displays sentiment analysis results in a bar chart.
- User-Friendly Interface: Utilizes Streamlit for easy interaction and visualization.

### Technologies Used

- Streamlit: For creating the interactive web interface
- NLTK: Natural Language Toolkit for text processing and sentiment analysis
- spaCy: Modern natural language understanding library
- scikit-learn: Machine learning algorithms for classification
- TextBlob: Simple API for diving into common NLP tasks
- Librosa: Audio analysis library
- Speech Recognition: For transcribing audio files

### Solution Structure

This Solution consists of several key components:

1. EnhancedAnalyzer Class: Centralizes all analysis functionality
2. Text Analysis Module: Handles sentiment scoring, subjectivity detection, and psychological state classification
3. Audio Processing Module: Transcribes audio and extracts tempo and pitch features
4. Visualization Module: Creates interactive charts using Plotly
5. Main Application: Manages user input selection and results display



### Solution Approach

I developed a web interface that takes user input in the form of text or audio files and outputs sentiment and psychological insights about the speakers. The application combines natural language processing, machine learning, and web development to provide a comprehensive analysis tool.

Key features of my implementation:

1. **Text Input**: Users can input text directly through a web interface, and the application analyzes sentiment, subjectivity, psychological state, and entities within the text.

2. **Audio Input**: Users can upload audio files, which are transcribed and analyzed for sentiment, subjectivity, psychological state, and audio features like tempo and pitch.

3. **Sentiment Analysis**: I implemented sentiment analysis using NLTK's VADER tool and TextBlob library.

4. **Psychological State Classification**: I developed a custom classification model using Random Forest to identify psychological states from text inputs.

5. **Entity Extraction**: I integrated spaCy for entity recognition in text inputs.

6. **Audio Processing**: I added functionality to analyze audio files using librosa, extracting tempo and pitch features.

7. **Web Interface**: I built the entire application using Streamlit, providing an interactive and user-friendly interface for both text and audio inputs.

### Challenges Faced

During development, I encountered several challenges:

1. **Data Collection**: Gathering sufficient training data for the psychological state classification model proved challenging. I had to manually create a dataset of sample phrases associated with different psychological states.

2. **Model Accuracy**: Achieving high accuracy in classifying psychological states was difficult due to the subjective nature of human emotions and the complexity of psychological states.

3. **Audio Transcription**: Implementing reliable speech-to-text transcription was tricky, especially with background noise or accents.

4. **Performance Optimization**: Balancing the trade-off between model performance and real-time processing speed was a challenge, particularly for longer texts or audio files.

5. **User Experience**: Designing an intuitive interface that guides users through the input selection and result interpretation was important but challenging.

### Running the Code

To run the application locally, follow these steps:

1. Ensure you have the required libraries installed:
   ```
   pip install streamlit nltk spacy scikit-learn librosa numpy plotly
   ```

2. Download the necessary NLTK resources:
   ```
   python -m nltk.downloader vader_lexicon punkt averaged_perceptron_tagger maxent_ne_chunker words
   ```

3. Install spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```

4. Clone the repository and navigate to the project directory:
   ```
   git clone https://github.com/Mihawk1891/SameyAI.git
   cd sentiment_analysis_app
   ```

5. Run the Streamlit app:
   ```
   streamlit run Problem_1.py
   ```

6. Open a web browser and navigate to http://localhost:8501 to access the application.



## • Rag (SameyRAG.ipynb)

### Solution Overview

This solution combines the strengths of traditional information retrieval systems with advanced language models. Our RAG implementation allows users to query a corpus of documents and receive responses generated by an LLM, augmented with relevant retrieved information.

Key features of this project:

- Supports both dense vector retrieval and BM25-based retrieval methods
- Integrates with Google Generative AI for LLM capabilities
- Implements a modular design for easy customization and experimentation
- Provides a chat interface for user interaction

### How It Works

My RAG system operates as follows:

1. **Document Loading**: PDF files are loaded and processed using PyPDFLoader and RecursiveCharacterTextSplitter.

2. **Vector Store Creation**: Documents are converted into numerical embeddings using GoogleGenerativeAIEmbeddings and stored in a FAISS vector store.

3. **Retriever Selection**: Users can choose between dense retrieval and BM25 retrieval methods.

4. **Query Processing**: When a user inputs a query, our system retrieves relevant documents from the chosen vector store.

5. **Response Generation**: The retrieved documents are passed along with the original query to the LLM (Google Generative AI).

6. **Response Formatting**: The final response is formatted according to the prompt template.

### Technical Details

- **Retrieval Methods**: 
  - Dense retrieval uses FAISS for efficient similarity searches.
  - BM25Okapi provides relevance-ranked document retrieval.

- **LLM Integration**: 
  - Utilizes ChatGoogleGenerativeAI for generating responses.
  - Temperature control is implemented for fine-tuning output quality.

- **Modular Design**: 
  - Separate functions for loading documents, creating vector stores, retrievers, and RAG chains.
  - Allows for easy modification and extension of individual components.



### Solution approach

### Understanding the Problem Statement

The objective of this project is to develop a system that retrieves relevant documents from a given corpus and uses them to generate responses. This task involves integrating two key components:

1. A retrieval method (such as BM25 or dense retrieval)
2. A Language Model (LLM) for generating responses

### Solution Approach

I approached this challenge by implementing a Retrieval-Augmented Generation (RAG) architecture. I designed a system that combines state-of-the-art retrieval techniques with advanced language generation capabilities.

### Retrieval Methods Implemented

I developed My system to support two retrieval methods:

1. Dense Retrieval: This method uses Google GenerativeAI embeddings to capture semantic similarities between documents and queries. It's particularly effective for large-scale datasets and can capture nuanced semantic relationships.

2. BM25 Retrieval: This term-frequency based approach is excellent for smaller datasets or when term frequencies are crucial. It allows for fine-tuning of weighting schemes to optimize retrieval performance.

### Integration with Language Model

My RAG chain integrates the retrieval method with the LLM in the following way:

1. I create a dynamic retriever function that allows us to switch between dense and BM25 retrieval methods based on the specific requirements of the task and corpus nature.

2. The retrieved relevant documents are then formatted and used as context for the LLM.

3. My prompt template guides the LLM to generate responses based on both the user input and the retrieved context.

### Technical Implementation Details

I implemented the following components:

1. Document Loading and Splitting: We developed functions to load PDF files and split them into manageable chunks for processing.

2. Vector Store Creation: We utilized FAISS vector store to efficiently manage and query document embeddings.

3. Retrieval Method Selection: Our system dynamically chooses between dense and BM25 retrieval methods based on user input.

4. RAG Chain Creation: We designed a flexible RAG chain structure that incorporates both retrieval and generation components.

5. Chat Loop Implementation: We developed a chat loop that continuously processes user queries and generates responses based on the retrieved information.

### Benefits of This Approach

This RAG architecture offers several advantages:

1. Flexibility: It allows for easy switching between dense and BM25 retrieval methods, adapting to different corpus characteristics.

2. Scalability: The system can handle large datasets efficiently due to the use of vector stores and optimized retrieval algorithms.

3. Contextual Understanding: By incorporating relevant documents into the LLM's context, we enhance the quality and accuracy of generated responses.

4. User-Friendly Interface: Our chat-based interface makes it easy for users to interact with the system and receive relevant information.

### Usage

1. Clone the repository:
   ```
   git clone https://github.com/Mihawk1891/SameyAI.git
   ```

2. Set up the Python environment:
   - Navigate to the `python_env` folder
   - Import the `NLP.yml` file to create a dedicated Python environment

3. Configure API keys:
   - Set your Google API key in the `os.environ["GOOGLE_API_KEY"]`
   - Add your OpenAI API key in the `common/openAI.env` file

4. Run the main SameyRAG.ipynb file:
   ```
   SameyRAG.ipynb
   ```
   This will start the chat interface where you can interact with the RAG system.


1. Run the main script to start the chat interface.
2. Type your queries or questions.
3. The system will retrieve relevant documents and generate responses based on those documents and your input.


## •  LLM based document analysis system (problem_3.py)

### Solution Overview

The project implements an optimized Large Language Model (LLM) based document analysis system. It leverages various techniques to improve performance, including hybrid vector search, semantic caching, dynamic batching, and potential embedding quantization. The system processes PDF documents, extracts relevant information, and provides detailed analysis based on user queries.

### Key Components

1. **Document Loading and Splitting**
   - Uses PyPDFLoader to load PDF documents
   - Employs RecursiveCharacterTextSplitter for efficient document splitting

2. **Embedding Generation**
   - Utilizes SentenceTransformer for generating high-quality sentence embeddings
   - Integrates Google Generative AI for text generation capabilities

3. **Vector Search Optimization**
   - Implements hybrid vector search using HNSW algorithm
   - Employs FAISS for efficient similarity search

4. **Query Processing and Batching**
   - Implements dynamic batching for asynchronous query processing
   - Uses asyncio for parallel execution of queries

5. **Semantic Caching**
   - Implements a caching mechanism using SentenceTransformer and FAISS
   - Stores and retrieves previously processed queries and their corresponding results

6. **Quantization of Embeddings**
   - Includes functions for quantizing and dequantizing embeddings
   - Potentially reduces memory usage and computational complexity

## Getting Started

To run this application locally:

1. Clone the repository `https://github.com/Mihawk1891/SameyAI.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables (API_KEY, etc.)
4. Run the server: `python problem_3.py`

To interact with the API:

- Send POST requests to `http://localhost:8000/query`
- Include query text in the request body

## My Solution approach for this problem 
 
### 1. Hybrid Vector Search with HNSW Algorithm

I developed this optimization technique using the Hierarchical Navigable Small World (HNSW) algorithm to efficiently search through large vector spaces.

I implemented this optimization by creating an HNSW index using the `hnswlib` library. I embedded documents using `GoogleGenerativeAIEmbeddings`. The `hybrid_search` function I implemented actually searches the HNSW index to quickly retrieve relevant documents based on similarity, improving overall system performance.

I chose this approach because the HNSW algorithm provides faster search times compared to traditional indexing methods like KD-trees or ball trees. It maintains a good balance between search speed and memory usage, making it suitable for large-scale applications.

### 2. Semantic Caching

I developed this optimization technique to cache frequently accessed results and reduce redundant computations.

I implemented this optimization by creating a `SemanticCache` class that manages the caching mechanism. I used Faiss' `IndexFlatIP` for efficient storage and retrieval of embeddings. The cache is dynamically updated after each successful search to ensure up-to-date results.

I chose this approach because caching allows me to avoid recalculating responses for repeated queries, improving response time for frequently asked questions. The dynamic adjustment of cache size prevents memory overflow.

### 3. Dynamic Batching

I developed this optimization technique to process queries in batches, optimizing API calls and resource utilization.

I implemented this optimization by creating a `QueryBatcher` class that handles asynchronous query processing. I collect queries in a queue and process them in batches. The `process_batch` method I implemented coordinates the execution of queries and updates the cache.

I chose this approach because batch processing reduces the number of API calls to the LLM, helps manage concurrent queries more efficiently, and reduces the overhead associated with individual query processing.

### 4. Quantization of Embeddings

I developed this optimization technique to compress high-dimensional embeddings, reducing memory usage and computation time.

I implemented this optimization by developing the `quantize_embeddings` function to perform the quantization process. I also created the `dequantize_embeddings` function to reverse the process when needed. This technique can be applied to both input and output embeddings.

I chose this approach because quantization significantly reduces the memory footprint of embeddings. It speeds up operations involving embeddings without sacrificing much accuracy, allowing for more efficient storage and transmission of embeddings.

These optimizations collectively contribute to improved performance by reducing computational complexity, enhancing search efficiency, managing resources effectively, and optimizing memory usage. They allow the system to handle larger datasets and more complex queries while maintaining responsiveness and accuracy.

### Performance Enhancements

1. **Parallel Processing**: Utilizes asyncio for asynchronous processing of queries and batching.

2. **Efficient Document Loading**: Employs RecursiveCharacterTextSplitter for efficient document splitting.

3. **Vector Retrieval Efficiency**: Leverages FAISS for efficient similarity search and HNSW algorithm for approximate nearest neighbor search.

4. **Reduced Redundancy**: Implements semantic caching to reduce redundant computations for repeated queries.

5. **Asynchronous Execution**: Uses asyncio to manage concurrent processing of queries and batches.
