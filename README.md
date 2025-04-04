# Conversational RAG with PDF Upload and Chat History

This is a Streamlit-based application that implements a **Conversational Retrieval-Augmented Generation (RAG)** system. It allows users to upload PDF files, ask questions about their content, and maintain a chat history for context-aware responses. The app uses Groq's language model for generation, FAISS for vector storage, and Hugging Face embeddings for text processing.

## Features
- Upload one or more PDF files to extract and process content.
- Ask questions about the PDF content via a chat interface.
- Maintains conversation history for context-aware responses.
- Uses Groq's `gemma2-9b-it` model for natural language generation.
- Vectorizes PDF content with FAISS and Hugging Face's `all-MiniLM-L6-v2` embeddings.

## Prerequisites
- Python 3.8 or higher
- A Groq API key (sign up at [Groq Console](https://console.groq.com))
- A Hugging Face API key (optional, for embeddings; set in `.env`)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ddv2311/Conversational-RAG-with-PDF-Upload.git
   cd Conversational-RAG-with-PDF-Upload

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   
3. **Install Dependencies and Run**
   ```bash
   pip install -r requirements.txt
   streamlit run app.py


   
