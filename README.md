# Document-Context-QNA-System
Created an LLM application using langchain and llama 3.1 for uploading xlsx, xls, csv, pdf and txt files and then answering questions based on them.
The streamlit code has been given in the repo

1. An Ollama instance and python installation should also be available for this project. Download and install ollama from [https://ollama.com/download](https://ollama.com/download) 
```sh
ollama serve
ollama pull llama3.1:8b
```
2. Run the Streamlit application:
```sh
pip install -r requirements.txt
streamlit run streamlit_app.py
```
