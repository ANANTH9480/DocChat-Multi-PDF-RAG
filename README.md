
# DocChat - Multi-PDF Chatbot using RAG

## Project Overview
DocChat is an AI-powered application that allows users to upload multiple PDF documents and ask questions about their content.

It uses Retrieval Augmented Generation (RAG) to retrieve relevant information and generate answers using a local Large Language Model.

## Features
- Upload multiple PDF documents.
- Extract text from PDFs.
- Split documents into smaller chunks.
- Generate text embeddings using Hugging Face.
- Store embeddings using FAISS.
- Retrieve relevant information from individual PDFs.
- Generate answers using Ollama.
- Compare resumes and provide estimated ratings.
- Display source PDF references and page numbers.
- Chat interface using Streamlit.

## Technologies Used
- Python
- Streamlit
- LangChain
- Hugging Face
- FAISS
- Ollama
- Qwen2.5 1.5B
- Sentence Transformers

## RAG Workflow
1. Document Loading
2. Text Chunking
3. Text Embeddings
4. Vector Database
5. Document Retrieval
6. Prompt Engineering
7. LLM Response Generation

## How to Run

Install dependencies:

pip install -r requirements.txt

Run Ollama with the required model:

ollama pull qwen2.5:1.5b

Start the application:

streamlit run app.py

## Author
Ananth

## Application Screenshots

### 1. Main Application
![DocChat Home](screenshots/01_home.png)

### 2. Resume Comparison using RAG
![Resume Comparison](screenshots/02_comparison.png)

### 3. Retrieved Source References
![Source References](screenshots/03_sources.png)
