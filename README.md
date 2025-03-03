# MedAI Chatbot

MedAI Chatbot is a medical AI-powered chatbot that uses natural language processing (NLP) techniques to answer user queries based on information extracted from PDF documents. It utilizes FAISS for vector embeddings and Mistral-7B for question answering.

![MedAI Chatbot Demo]('/Users/dharyavardhan/Desktop/Screenshots/Screenshot 2025-03-03 at 9.29.32 PM.png')

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Variables](#environment-variables)

## Features
- Extracts text from PDFs and converts them into vector embeddings.
- Stores and retrieves vector embeddings using FAISS.
- Uses Mistral-7B model from Hugging Face for answering queries.
- Implements a Streamlit-based user interface for interaction.
- Ensures reliable and accurate responses within a given document context.

## Technologies Used
- **Python** (Main programming language)
- **LangChain** (For document processing and LLM integration)
- **FAISS** (For efficient similarity search and vector retrieval)
- **Sentence Transformers (all-MiniLM-L12-v2)** (For generating text embeddings)
- **Streamlit** (For the chatbot UI)
- **Hugging Face Transformers** (For the LLM integration)
- **dotenv** (For environment variable management)

## Models Used
- **Embedding Model**: `sentence-transformers/all-MiniLM-L12-v2`
- **LLM Model**: `mistralai/Mistral-7B-Instruct-v0.3`
  - Initially tried `LLaMA 3` and other similar models, but Mistral-7B provided better results in terms of accuracy and response quality.

## Project Structure
```
MedAI-Chatbot/
│── data/                      # Folder containing PDF files
│── vectorstore/               # FAISS vector database
│── main.py                    # Streamlit application
│── pdf_loader.py              # PDF processing and vector embedding
│── chatbot.py                 # Chatbot logic and model integration
│── requirements.txt           # Required dependencies
│── .env                       # Environment variables (API keys, etc.)
│── README.md                  # Documentation
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/MedAI-Chatbot.git
cd MedAI-Chatbot
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file and add the following:
```
HF_TOKEN=your_huggingface_api_token
```

## Usage

### Step 1: Process PDFs and Generate Embeddings
Run the following command to process PDF files and store them as vector embeddings:
```bash
python pdf_loader.py
```

### Step 2: Start the Chatbot
Run the Streamlit chatbot application:
```bash
streamlit run main.py
```

### Step 3: Interact with the Chatbot
- Enter a query related to the uploaded PDF documents.
- The chatbot retrieves relevant information from the documents and generates an accurate response.

## Environment Variables
Ensure the following environment variables are set in your `.env` file:
- `HF_TOKEN` – Your Hugging Face API token for accessing the Mistral-7B model.


