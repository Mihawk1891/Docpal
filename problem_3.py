import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document

app = FastAPI()

os.environ["GOOGLE_API_KEY"] = "API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Optimization 1: Hybrid Vector Search with HNSW Algorithm
def create_hybrid_vector_store(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = embeddings.embed_documents([doc.page_content for doc in documents])

    dim = len(vectors[0])
    num_elements = len(vectors)

    hnsw_index = hnswlib.Index(space='cosine', dim=dim)
    hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=16)
    hnsw_index.add_items(vectors, list(range(num_elements)))

    return hnsw_index, documents, embeddings

def hybrid_search(hnsw_index, documents, embeddings, query, k=5):
    query_vector = embeddings.embed_query(query)
    labels, distances = hnsw_index.knn_query(query_vector, k)
    return [documents[label] for label in labels[0]]

# Optimization 2: Semantic Caching
class SemanticCache:
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_size=1000):
        self.model = SentenceTransformer(model_name)
        self.cache_size = cache_size
        self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
        self.queries = []
        self.results = []

    def get(self, query):
        query_embedding = self.model.encode([query])[0]
        if self.index.ntotal > 0:
            D, I = self.index.search(query_embedding.reshape(1, -1), 1)
            if D[0][0] > 0.9:  # Similarity threshold
                return self.results[I[0][0]]
        return None

    def add(self, query, result):
        if self.index.ntotal >= self.cache_size:
            self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
            self.queries = []
            self.results = []
        query_embedding = self.model.encode([query])[0]
        self.index.add(query_embedding.reshape(1, -1))
        self.queries.append(query)
        self.results.append(result)

semantic_cache = SemanticCache()

# Optimization 3: Dynamic Batching
class QueryBatcher:
    def __init__(self, batch_size=5, max_wait_time=0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.query_queue = asyncio.Queue()
        self.result_futures = {}

    async def add_query(self, query: str):
        future = asyncio.Future()
        await self.query_queue.put((query, future))
        return await future

    async def process_batches(self):
        while True:
            batch = []
            batch_futures = []
            try:
                async with asyncio.timeout(self.max_wait_time):
                    while len(batch) < self.batch_size:
                        query, future = await self.query_queue.get()
                        batch.append(query)
                        batch_futures.append(future)
            except asyncio.TimeoutError:
                pass

            if batch:
                results = await self.process_batch(batch)
                for future, result in zip(batch_futures, results):
                    future.set_result(result)

    async def process_batch(self, queries: List[str]):
        # Implement batch processing logic here
        results = []
        for query in queries:
            cached_result = semantic_cache.get(query)
            if cached_result:
                results.append(cached_result)
            else:
                result = hybrid_search(hnsw_index, documents, embeddings, query)
                semantic_cache.add(query, result)
                results.append(result)
        return results

query_batcher = QueryBatcher()

# Optimization 4: Quantization of Embeddings
def quantize_embeddings(embeddings, bits=8):
    min_val = np.min(embeddings)
    max_val = np.max(embeddings)
    scale = (max_val - min_val) / (2**bits - 1)
    quantized = np.round((embeddings - min_val) / scale).astype(np.uint8)
    return quantized, min_val, scale

def dequantize_embeddings(quantized, min_val, scale):
    return quantized.astype(np.float32) * scale + min_val

# Main functions
def load_and_split_documents(file_paths):
    documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

prompt_template = """
User Query: {user_input}

Relevant Corpus Data:
{context}

You are a document analysis assistant. Based on the User Query and the relevant Corpus data, please provide a detailed and accurate response. If you need any clarification or additional information, please ask.

The answer should be in points and then subpoints. Use paragraphs only when necessary.

Focus solely on the document content to answer the user's question. If there is a user query that cannot be answered using the provided context, respond with 'Please ask questions about the Corpus'.

Do not repeat the user's question. If the user's query is vague, provide answers and also suggest more specific questions.

Chat History:
{chat_history}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

async def async_llm(prompt, llm):
    return await asyncio.to_thread(llm, prompt)

class Query(BaseModel):
    text: str

@app.post("/query")
async def process_query(query: Query):
    relevant_docs = await query_batcher.add_query(query.text)

    context = format_docs(relevant_docs)
    prompt = ChatPromptTemplate.from_template(prompt_template).format(
        user_input=query.text, context=context, chat_history=""
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    response = await async_llm(prompt, llm)

    return {"response": StrOutputParser().parse(response)}

# Initialization
file_paths = ["Corpus.pdf"]  # Add more file paths as needed
documents = load_and_split_documents(file_paths)
hnsw_index, documents, embeddings = create_hybrid_vector_store(documents)

# Start the query batcher
batcher_task = None

async def start_batcher():
    global batcher_task
    batcher_task = asyncio.create_task(query_batcher.process_batches())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global batcher_task
    batcher_task = asyncio.create_task(query_batcher.process_batches())
    yield
    # Shutdown
    if batcher_task:
        batcher_task.cancel()
        try:
            await batcher_task
        except asyncio.CancelledError:
            pass

app = FastAPI(lifespan=lifespan)

# Initialization
file_paths = ["Corpus.pdf"]  # Add more file paths as needed
documents = load_and_split_documents(file_paths)
hnsw_index, documents, embeddings = create_hybrid_vector_store(documents)

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()

import asyncio
from rag_server import run_server

# Run the server in a separate thread
import threading
server_thread = threading.Thread(target=run_server)
server_thread.start()

import requests

def query(text):
    response = requests.post("http://localhost:8000/query", json={"text": text})
    return response.json()

# I am using this Query for example
print(query("What is the main topic of the document?"))

# When you're done, you can stop the server by interrupting the kernel