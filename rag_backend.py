import os
import tempfile
from typing import List, Tuple, Any

# PDF Loading
from langchain_community.document_loaders import PyPDFLoader

# Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector Store
from langchain_community.vectorstores import FAISS

# Multi-Query Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever

# Cross-Encoder based Reranking
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Prompt Template
from langchain.prompts import PromptTemplate

# LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Memory & Chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document, AIMessage, HumanMessage # Ensure Document is imported

# Optional: Load environment variables if you're using a .env file
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash" 

def load_pdfs_with_pypdf(uploaded_files: List[Any]) -> List[Document]:
    """
    Loads PDF files using PyPDFLoader and returns a list of Langchain Document objects.
    Each UploadedFile is saved to a temporary file to be processed by PyPDFLoader.
    """
    documents = []
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                tmp_file_path = tmpfile.name
            
            # Use the original filename in metadata if possible
            loader = PyPDFLoader(tmp_file_path)
            # PyPDFLoader adds metadata like source (temp path) and page number
            docs_for_file = loader.load()
            
            # Update metadata to show original filename instead of temp path
            for doc in docs_for_file:
                doc.metadata['source'] = uploaded_file.name # Override temp path with original name
            documents.extend(docs_for_file)

            os.remove(tmp_file_path)
        except Exception as e:
            print(f"Error loading PDF {uploaded_file.name}: {e}")
    return documents

def get_document_chunks(documents: List[Document]) -> List[Document]:
    """
    Splits a list of Langchain Document objects into smaller, overlapping chunks.

    Args:
        documents: A list of Langchain Document objects.

    Returns:
        A list of chunked Langchain Document objects.
    """
    if not documents:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        # length_function=len # Default is len, explicitly stating for clarity
    )
    # split_documents directly works with Document objects
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

def get_vector_store(document_chunks: List[Document]):
    """
    Creates a FAISS vector store from document chunks using HuggingFace embeddings.

    Args:
        document_chunks: A list of chunked Langchain Document objects.

    Returns:
        A FAISS vector store object, or None if no chunks are provided or an error occurs.
    """
    if not document_chunks:
        print("No document chunks to process for vector store.")
        return None
    try:
        print(f"Initializing HuggingFace Embeddings with model: {EMBEDDING_MODEL_NAME}")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'} # Or 'cuda' if GPU is available and configured
        )
        print("HuggingFace Embeddings initialized.")

        print(f"Creating FAISS vector store from {len(document_chunks)} document chunks...")
        # FAISS.from_documents expects a list of Document objects and an Embedding object
        vectorstore = FAISS.from_documents(documents=document_chunks, embedding=embeddings)
        print("FAISS vector store created successfully.")
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        # Consider raising the exception or returning None to be handled by the caller
        raise

def get_conversational_chain(vector_store):
    """
    Creates a conversational retrieval chain using Google Gemini.

    Args:
        vector_store: The FAISS vector store.

    Returns:
        A ConversationalRetrievalChain object.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it to use Gemini.")

    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=0.1,
            # convert_system_message_to_human=True # May be needed for some chains/models if system prompts are used heavily
        )
    except Exception as e:
        print(f"Error initializing Gemini LLM: {e}")
        raise

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4}), llm=llm)
    
    
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=multi_query_retriever)

    prompt = PromptTemplate(template = """
                          You are a helpful Legal Assistant.
                          Answer ONLY from the provided transcript context. Describe it in a simple way.
                          If the context is insufficient, just say you don't know.
                          {context}
                          Question: {question}
                          """,
                          input_variables= ['context', 'question'])
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        memory=memory,
        return_source_documents=True, # Optionally return source documents
        output_key='answer'
    )
    return conversation_chain

def handle_chat_interaction(user_question: str, conversation_chain, chat_history: List[Tuple[str, str]]):
    """
    Handles a single user interaction, gets a response from the conversational chain.

    Args:
        user_question: The user's current question.
        conversation_chain: The ConversationalRetrievalChain instance.
        chat_history: The current chat history (list of HumanMessage, AIMessage).

    Returns:
        - The AI's response string.
        - A list of source Document objects.
    """
    try:
        response = conversation_chain.invoke({
            "question": user_question,
            "chat_history": chat_history # The chain uses its memory, but this can provide explicit context for the turn
        })
        return response['answer'], response.get('source_documents', [])
    except Exception as e:
        print(f"Error during chat interaction with chain: {e}")
        # Depending on the error, you might want to return a user-friendly message
        # For instance, if it's an API error from Gemini
        if "API key not valid" in str(e) or "permission" in str(e).lower():
            return "Error: Could not connect to the AI model. Please check the API key and permissions."
        return "Sorry, I encountered an error while processing your request."

