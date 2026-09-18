# 📚 RAG PDF Chatbot

A simple Retrieval-Augmented Generation (RAG) based PDF chatbot built with Python, Streamlit, LangChain, Google Gemini, and FAISS.

The application allows users to upload a PDF document and ask questions about its content. The PDF text is extracted, divided into smaller chunks, converted into embeddings, and stored in a FAISS vector database. When a user asks a question, relevant chunks are retrieved and provided to Gemini to generate the answer.

## 🚀 Features

- 📄 Upload PDF documents
- 📝 Extract text from PDF
- ✂️ Split large text into smaller chunks
- 🧠 Generate embeddings using Google Gemini
- 🔎 Perform similarity search using FAISS
- 🤖 Generate answers using Gemini
- 💻 Simple and interactive Streamlit interface
- 🔐 API key stored securely using environment variables

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini
- FAISS
- PyPDF2
- python-dotenv

## 🔄 How It Works

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Gemini Embeddings
    ↓
FAISS Vector Database
    ↓
Similarity Search
    ↓
Relevant Document Chunks
    ↓
Gemini LLM
    ↓
Answer
