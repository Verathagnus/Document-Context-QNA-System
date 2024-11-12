# Document-Context-QnA-System

This project is a **Conversational Document-Based QnA System** that allows users to upload various document types, including XLSX, CSV, PDF, and TXT, and then answer questions based on the contents of these documents. Leveraging LangChain, LLaMA 3.1, and Streamlit, this system is built for interactive question-answering using custom document data.

## Features

- **Multi-Format Document Support**: Accepts XLSX, CSV, PDF, and TXT formats for a wide range of document compatibility.
- **RAG (Retrieval-Augmented Generation)**: Retrieves relevant document sections to generate contextually accurate answers.
- **Ollama LLaMA 3.1 Integration**: Uses LLaMA 3.1 8B as the main model for both embeddings and response generation.
- **User-Friendly Streamlit Interface**: Simple UI for document uploading, question input, and chat history.

## Setup and Requirements

### Step 1: Clone this Repository

```sh
git clone https://github.com/yourusername/Document-Context-QnA-System.git
cd Document-Context-QnA-System
```

### Step 2: Install Ollama and Prepare the LLaMA Model

1. Download and install Ollama from [ollama.com/download](https://ollama.com/download).
2. Start the Ollama instance:

   ```sh
   ollama serve
   ollama pull llama3.1:8b
   ```

### Step 3: Install Project Dependencies

```sh
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```sh
streamlit run streamlit_app.py
```

## Workflow

### 1. **Document Upload and Parsing**

   - The user uploads one or more documents (CSV, XLSX, PDF, or TXT) through the Streamlit interface.
   - The application parses and processes each document, splitting them into manageable chunks for efficient storage and retrieval.

### 2. **Storing Documents in Vector Database**

   - Each chunk of the document is embedded using LLaMA 3.1 embeddings and stored in ChromaDB.
   - The embeddings are used to index the content, enabling fast and relevant retrieval based on user queries.

### 3. **QnA Chain Initialization**

   - The system initializes a QnA chain if one hasn’t been set up yet, combining retrieval and response-generation capabilities.
   - This chain will retrieve contextually relevant sections of the document and pass them as a prompt to the language model for response generation.

### 4. **User Query and Document Retrieval**

   - The user enters a question, which is used to retrieve the top `k` most relevant document chunks from ChromaDB (default `k=20`).
   - These retrieved document sections are structured into a RAG prompt, combining the question and context.

### 5. **Answer Generation and Display**

   - The prompt is passed to the `ChatOllama` model to generate a concise answer based on both the question and document context.
   - The answer is displayed to the user in the Streamlit chat interface.
   - The conversation history (questions and answers) is displayed to provide continuity and context for follow-up questions.

### Streamlit Workflow Management

   - `st.session_state` is used to track the state of vector storage, processing status, and chat history.
   - A loading spinner provides feedback while processing documents or generating answers.
   - A “Clear All” button allows users to reset the entire session, clearing vector storage and conversation history.

## Usage

- **Document Upload**: Upload documents in XLSX, CSV, PDF, or TXT format for parsing and storage in ChromaDB.
- **Question Input**: Enter questions related to the uploaded documents and receive contextually accurate answers.
- **Context Display**: View the conversation history to keep track of all questions and responses.

---
