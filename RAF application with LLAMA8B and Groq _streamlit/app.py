import streamlit as st
import os
import pickle
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_groq import ChatGroq

# Filepath to save embeddings
embedding_file = "document_embeddings.pkl"
api_key = "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"  # Replace with your actual API key
llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=api_key)

def load_pdf(uploaded_file):
    """Load the PDF file."""
    start_time = time.time()
    with open("temp_pdf_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    docs = PyPDFLoader("temp_pdf_file.pdf").load()
    end_time = time.time()
    st.write(f"PDF loading time: {end_time - start_time:.2f} seconds")
    return docs

def split_document(docs):
    """Split the document into chunks."""
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    end_time = time.time()
    st.write(f"Document splitting time: {end_time - start_time:.2f} seconds")
    return splits

def create_embeddings(splits):
    """Create and save embeddings if no existing embeddings are found, with a progress bar."""
    start_time = time.time()

    # Remove old embedding file if it exists
    if os.path.exists(embedding_file):
        st.write("Removing old embeddings file...")
        os.remove(embedding_file)

    st.write("Embedding the document...")

    # Setup progress bar
    progress_bar = st.progress(0)
    total_chunks = len(splits)

    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    
    hf_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    # Initialize an empty list to hold the documents and their embeddings
    embedded_docs = []

    # Embed chunks one by one and update progress bar
    for i, split in enumerate(splits):
        embedded_docs.append(split)  # Add the split to the list for embedding

        # Update the progress bar
        progress_percentage = (i + 1) / total_chunks
        progress_bar.progress(progress_percentage)

    # Create FAISS vectorstore after all chunks are embedded
    vectorstore = FAISS.from_documents(embedded_docs, hf_embeddings)

    # Save embeddings to disk
    with open(embedding_file, "wb") as f:
        pickle.dump(vectorstore, f)
    
    end_time = time.time()
    st.write(f"Embedding time: {end_time - start_time:.2f} seconds")
    return vectorstore




def load_embeddings():
    """Load embeddings from disk if they exist."""
    if os.path.exists(embedding_file):
        st.write("Loading existing embeddings from disk...")
        with open(embedding_file, "rb") as f:
            vectorstore = pickle.load(f)
        return vectorstore
    else:
        st.error("Embeddings not found. Please upload a document first.")
        return None

def setup_rag_chain(vectorstore):
    """Set up the RAG pipeline."""
    start_time = time.time()
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    end_time = time.time()
    st.write(f"RAG chain setup time: {end_time - start_time:.2f} seconds")
    return rag_chain

def main():
    st.title("RAG on PDF Document")

    # Step 1: Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        st.write("Processing the PDF...")
        
        # Step 2: Load and Split the PDF Document
        docs = load_pdf(uploaded_file)
        splits = split_document(docs)
        st.write(f"Document split into {len(splits)} chunks.")
        
        # Step 3: Create Embeddings
        vectorstore = create_embeddings(splits)
        
        # Set up RAG Chain after embeddings are created
        st.session_state["rag_chain"] = setup_rag_chain(vectorstore)

    # Step 4: Load existing embeddings (if no new PDF is uploaded)
    else:
        vectorstore = load_embeddings()
        if vectorstore:
            if "rag_chain" not in st.session_state:
                st.session_state["rag_chain"] = setup_rag_chain(vectorstore)

    # Step 5: Input and Output
    if "rag_chain" in st.session_state:
        query = st.text_input("Ask a question about the document:")
        if query:
            st.write("Querying the document...")
            start_time = time.time()
            answer = st.session_state["rag_chain"].invoke(query)
            end_time = time.time()
            st.write(f"Querying time: {end_time - start_time:.2f} seconds")
            st.write("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
