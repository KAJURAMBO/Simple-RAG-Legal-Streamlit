import streamlit as st
import os
import uuid
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
import time

# Set up working directory
folder_path = "./local_vectors"
os.makedirs(folder_path, exist_ok=True)

# Load pre-trained smaller embedding model and tokenizer from Hugging Face
tokenizer_model_name = "distilbert-base-uncased"  # Smaller model for tokenization and embeddings
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
model = AutoModel.from_pretrained(tokenizer_model_name)
API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
headers = {"Authorization": "Bearer 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"}

# Function to get embeddings from Hugging Face model
@st.cache_data
def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Pooling strategy
    return embeddings

# Function to get a unique request ID
def get_unique_id():
    return str(uuid.uuid4())

# Load the PDF and process it
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# Split the text into chunks
def split_text(pages, chunk_size=100, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

# Create vector store and save locally

@st.cache_data
def create_vector_store(request_id, _documents):
    start_time = time.time()
    documents = _documents  # Unwrap documents from cached argument
    # Create a Streamlit progress bar
    progress_bar = st.progress(0)
    
    # Get embeddings for the document texts
    texts = [doc.page_content for doc in documents]
    num_docs = len(texts)
    embeddings_list = []
    
    for i in range(0, num_docs, 50):  # Process 50 documents at a time
        batch_texts = texts[i:i + 50]
        embeddings = get_embeddings(batch_texts)
        embeddings_list.append(embeddings.numpy())  # Convert to numpy array
        
        # Update progress bar
        progress = (i + 50) / num_docs
        progress_bar.progress(min(progress, 1.0))
    
    # Ensure progress bar reaches 100%
    progress_bar.progress(1.0)
    
    # Concatenate all embeddings
    embeddings_np = np.vstack(embeddings_list)
    dimension = embeddings_np.shape[1]
    
    # Create a FAISS index
    index = faiss.IndexFlatL2(dimension)  # Use L2 distance (Euclidean)
    index.add(embeddings_np)  # Add embeddings to the index
    
    # Save the FAISS index locally
    file_name = f"{request_id}.faiss"
    faiss.write_index(index, os.path.join(folder_path, file_name))
    
    # Save the document metadata
    with open(os.path.join(folder_path, f"{request_id}.pkl"), 'wb') as f:
        pickle.dump(documents, f)
    end_time = time.time()
    st.write(f"Vector store created in {end_time - start_time:.2f} seconds.")
    return True


# Function to load the vector store from local files
@st.cache_data
def load_vector_store(request_id):
    start_time = time.time()
    
    faiss_path = os.path.join(folder_path, f"{request_id}.faiss")
    pkl_path = os.path.join(folder_path, f"{request_id}.pkl")
    
    # Load the FAISS index
    index = faiss.read_index(faiss_path)
    
    # Load document metadata
    with open(pkl_path, 'rb') as f:
        documents = pickle.load(f)
    end_time = time.time()
    st.write(f"Loading completed in {end_time - start_time:.2f} seconds.")
    
    return index, documents

# Custom FAISS retriever
class CustomFAISSRetriever:
    def __init__(self, index, documents):
        self.index = index
        self.documents = documents
    
    def get_relevant_documents(self, query):
        # Embed the query
        query_embedding = get_embeddings([query]).numpy()
        
        # Perform search
        distances, indices = self.index.search(query_embedding, k=5)  # Adjust k as needed
        
        # Retrieve documents
        relevant_docs = [self.documents[i] for i in indices[0]]
        return relevant_docs

# Function to query Hugging Face API for LLM inference
def query_huggingface_api(context, question, max_tokens=1024, max_new_tokens=200):
    start_time = time.time()
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    
    # Tokenize the input text
    input_tokens = tokenizer(input_text, return_tensors="pt")["input_ids"]
    total_tokens = input_tokens.shape[1]  # Number of tokens in input text
    
    # Truncate the context if it exceeds the token limit
    if total_tokens + max_new_tokens > max_tokens:
        # Calculate the maximum context length
        max_context_tokens = max_tokens - max_new_tokens
        truncated_input_tokens = input_tokens[0, :max_context_tokens]
        truncated_input_text = tokenizer.decode(truncated_input_tokens, skip_special_tokens=True)
        input_text = f"Context: {truncated_input_text}\nQuestion: {question}\nAnswer:"

    payload = {
        "inputs": input_text,
        "parameters": {"max_new_tokens": max_new_tokens, "do_sample": False},
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        response_json = response.json()
        if isinstance(response_json, list):
            answer = response_json[0].get("generated_text", "No text returned.")
        elif isinstance(response_json, dict):
            answer = response_json.get("generated_text", "No text returned.")
        else:
            answer = "Unexpected response format."
        
        # Clean the answer to avoid repeating context
        if answer:
            # Remove any extra context or repetitions
            cleaned_answer = answer.replace(input_text, "").strip()
            
            # Additional step to remove repeated phrases
            if cleaned_answer:
                parts = cleaned_answer.split('. ')
                unique_parts = list(dict.fromkeys(parts))  # Remove duplicates while preserving order
                final_answer = '. '.join(unique_parts).strip()
                
                return final_answer if final_answer else "No valid answer found."
            else:
                return "No valid answer found."
        else:
            return "No valid answer found."
    elif response.status_code == 503:
        return "Model is still loading, please try again later."
    else:
        return f"Error in API request: {response.status_code}, {response.text}"
    end_time = time.time()
    st.write(f"Inference completed in {end_time - start_time:.2f} seconds.")

# Streamlit app
def main():
    st.title("RAG with Hugging Face Open Embeddings")

    # Initialize session state
    if 'vector_store_created' not in st.session_state:
        st.session_state.vector_store_created = False
        st.session_state.request_id = None

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None and not st.session_state.vector_store_created:
        request_id = get_unique_id()
        st.session_state.request_id = request_id
        saved_file_name = f"{request_id}.pdf"

        # Save uploaded PDF
        with open(saved_file_name, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load and split the PDF
        pages = load_pdf(saved_file_name)
        st.write(f"Total Pages: {len(pages)}")

        # Split the text into chunks
        docs = split_text(pages)
        st.write(f"Total Chunks: {len(docs)}")

        # Create and save vector store locally
        st.write("Creating vector store...")
        success = create_vector_store(request_id, docs)
        if success:
            st.write("Vector store created and saved locally.")
            st.session_state.vector_store_created = True

    # User input for question
    if st.session_state.vector_store_created:
        question = st.text_input("Ask a question based on the PDF:")
        if question:
            # Load vector store from local files
            st.write("Loading vector store...")
            index, documents = load_vector_store(st.session_state.request_id)
            
            # Create custom FAISS retriever
            retriever = CustomFAISSRetriever(index, documents)
            relevant_docs = retriever.get_relevant_documents(question)
            context = " ".join([doc.page_content for doc in relevant_docs])
            
            # Call Hugging Face API for LLM inference
            st.write("Querying LLM via Hugging Face API...")
            response = query_huggingface_api(context, question)
            
            # Display the response
            st.write("Response:")
            st.write(response)

if __name__ == "__main__":
    main()