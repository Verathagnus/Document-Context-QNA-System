import streamlit as st
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatOllama
from langchain import hub
import io
import tempfile
import os

# Clear ChromaDB cache if needed
chromadb.api.client.SharedSystemClient.clear_system_cache()

# Initialize embeddings and Chroma vector store
embedding = OllamaEmbeddings(model="llama3.1:8b")
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Initialize Streamlit session state for vector stores and conversation
st.session_state.vector_store = Chroma(embedding_function=embedding)
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "question" not in st.session_state:
    st.session_state.question = ""  # To hold the current question
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False  # Track if a question is being processed

# Function to load and parse different file types from in-memory data
# Function to load and parse different file types from in-memory data
def load_documents(file):
    if file.name.endswith('.csv'):
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
            temp_csv.write(file.read())  # Write bytes to temp file
            temp_csv_path = temp_csv.name  # Get the file path for CSVLoader
        loader = CSVLoader(temp_csv_path)  # Pass the temporary file path to CSVLoader
        documents = loader.load()
        os.remove(temp_csv_path)  # Clean up temp file after loading
        return documents
    elif file.name.endswith('.xlsx'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_xlsx:
            temp_xlsx.write(file.read())  # Write bytes to temp file
            temp_xlsx_path = temp_xlsx.name  # Get the file path for CSVLoader
        loader = UnstructuredExcelLoader(temp_xlsx_path)  # Pass the temporary file path to CSVLoader
        documents = loader.load()
        os.remove(temp_xlsx_path)  # Clean up temp file after loading
        return documents
    elif file.name.endswith('.xls'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xls") as temp_xlsx:
            temp_xlsx.write(file.read())  # Write bytes to temp file
            temp_xlsx_path = temp_xlsx.name  # Get the file path for CSVLoader
        loader = UnstructuredExcelLoader(temp_xlsx_path)  # Pass the temporary file path to CSVLoader
        documents = loader.load()
        os.remove(temp_xlsx_path)  # Clean up temp file after loading
        return documents
    elif file.name.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file.read())  # Write bytes to temp file
            temp_pdf_path = temp_pdf.name
        loader = PyMuPDFLoader(temp_pdf_path)  # Load using file path
        documents = loader.load()
        os.remove(temp_pdf_path)  # Clean up temp file
        return documents
    elif file.name.endswith('.txt'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_txt:
            temp_txt.write(file.read())  # Write bytes to temp file
            temp_txt_path = temp_txt.name  # Get the file path for CSVLoader
        loader = TextLoader(temp_txt_path)  # Pass the temporary file path to CSVLoader
        documents = loader.load()
        os.remove(temp_txt_path) 
        return documents
    else:
        raise ValueError("Unsupported file type")
    return loader.load()


# Split documents into chunks for embedding
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Process and store documents in ChromaDB
def process_and_store_documents(files):
    for file in files:
        documents = load_documents(file)
        split_docs = split_documents(documents)
        st.session_state.vector_store.add_documents(split_docs)

# Initialize the QnA chain if not already initialized
def initialize_qa_chain():
    if not st.session_state.qa_chain:
        llm = ChatOllama(model="llama3.1:8b")
        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        retrieval_chain = create_retrieval_chain(
            st.session_state.vector_store.as_retriever(search_kwargs={"k": 20}),
            combine_docs_chain
        )
        st.session_state.qa_chain = retrieval_chain

# Function to run the QnA system
def ask_question(question):
    answer = st.session_state.qa_chain.invoke({"input": question})
    st.session_state.conversation.append((question, answer))  # Append question-answer pair to conversation
    return answer

# Function to clear all session state
def clear_all():
    # Clear conversation history and reset all necessary session states
    st.session_state.conversation = []
    st.session_state.vector_store = Chroma(embedding_function=embedding)  # Reinitialize vector store
    st.session_state.qa_chain = None
    st.session_state.is_processing = False
    st.session_state.question = ""
clear_button = st.button("Clear All")
if clear_button:
    clear_all()
    st.success("All data cleared.")
# Streamlit interface
st.title("Conversational Document-Based QnA System")

# File upload form
with st.form("my-form", clear_on_submit=True):
    file = st.file_uploader("Upload your documents (CSV, XLSX, PDF, TXT)", accept_multiple_files=True)
    submitted = st.form_submit_button("UPLOAD!")

    if submitted and file is not None:
        with st.spinner("Processing documents..."):
            process_and_store_documents(file)  # Use the file only here
            initialize_qa_chain()  # Initialize QnA chain
        st.success("Documents uploaded and processed successfully. You can now ask questions.")

# Add the "Clear All" button above the question input


# Chat interface for asking questions
st.subheader("Chat with Your Documents")

# Display conversation history
for q, a in st.session_state.conversation:
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a['answer']}")

# User input for new questions
question = st.text_input("Enter your question:", key="user_question_input", placeholder="Type your question here...")

# Add a button to send the message
send_button = st.button("Send Question", disabled=st.session_state.is_processing)

# Process the question only when the button is clicked and no other question is being processed
if send_button and not st.session_state.is_processing and question:
    st.session_state.is_processing = True  # Mark the system as processing
    with st.spinner("Sending your question..."):
        st.session_state.question = question  # Store the current question in session state
        initialize_qa_chain()
        answer = ask_question(question)
        st.session_state.is_processing = False  # Reset processing state
        st.session_state.question = ""  # Clear the question after processing
        st.rerun()

