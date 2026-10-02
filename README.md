# 📚 PDF RAG Assistant

A PDF-based Retrieval-Augmented Generation (RAG) question-answering system built using Python, Streamlit, Sentence Transformers, FAISS, PyMuPDF, and Groq.

## 🎯 Objective

The application allows users to upload PDF documents, ask questions about their contents, retrieve relevant information using semantic search, and generate context-aware answers using a Groq LLM.

## 🛠️ Technologies Used

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- Groq API
- LangChain Text Splitters
- python-dotenv

## 🔄 RAG Pipeline

PDF Upload
↓
Text Extraction
↓
Text Chunking
↓
Embeddings
↓
FAISS Vector Database
↓
Semantic Search
↓
Relevant Context
↓
Groq LLM
↓
Final Answer + Sources

## ✨ Features

- Upload PDF documents
- Extract text from PDF pages
- Split documents into smaller chunks
- Generate sentence embeddings
- Store embeddings using FAISS
- Perform semantic similarity search
- Configure Top-K retrieval
- Ask questions through a chatbot interface
- Generate answers using Groq
- Display document and page sources
- Maintain chat history during the session
- Clear chat history
- Handle PDFs with no readable text
- Avoid generating answers when information is unavailable in the retrieved context

## 📂 Project Structure

```text
pdf-rag-assistant_day 4/
│
├── app.py
├── README.md
├── .env
├── .gitignore
├── documents/
│   └── sample PDFs
└── venv/