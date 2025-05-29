import streamlit as st
import torch
import traceback
from rag_backend import (
    load_pdfs_with_pypdf, # Changed from load_pdfs_text
    get_document_chunks,
    get_vector_store,
    get_conversational_chain,
    handle_chat_interaction
)
from langchain.schema import HumanMessage, AIMessage
import os

torch.classes.__path__ = [os.path.join(torch.__path__[0], 'torch', 'classes')]

# Optional: Load .env file if it exists for local development
from dotenv import load_dotenv
load_dotenv() # Load GOOGLE_API_KEY from .env if present

def main():
    st.set_page_config(page_title="Conversational PDF Chat (Gemini + Sources) 📄", layout="wide")

    st.markdown("""
        <style>
        /* ... [your existing CSS styles] ... */
        .stApp { }
        .stButton>button { width: 100%; border-radius: 10px; }
        .stTextInput>div>div>input { border-radius: 10px; }
        .chat-bubble { padding: 10px; border-radius: 15px; margin-bottom: 10px; max-width: 70%; word-wrap: break-word; }
        .user-bubble { background-color: #DCF8C6; align-self: flex-end; margin-left: auto; }
        .ai-bubble { background-color: #E0E0E0; align-self: flex-start; margin-right: auto; }
        .sidebar .stFileUploader { }
        .sidebar .stButton>button { background-color: #4285F4; color: white; }
        .stExpander { border: 1px solid #ddd; border-radius: 10px; }
        .stExpanderHeader { font-size: 0.9rem; }
        .source-document { padding: 8px; margin-bottom: 5px; border-radius: 5px; background-color: #f9f9f9; }
        .source-document strong { font-size: 0.85rem; }
        .source-document p { font-size: 0.8rem; margin-bottom: 3px;}
        .source-document .metadata { font-size: 0.75rem; color: #555; margin-bottom: 5px;}
        </style>
    """, unsafe_allow_html=True)

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        # Store AIMessage objects, potentially with source_documents in additional_kwargs
        st.session_state.chat_history = []
    if "pdfs_processed" not in st.session_state:
        st.session_state.pdfs_processed = False
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []

    with st.sidebar:
        st.subheader("📚 Your Documents")
        uploaded_pdf_files = st.file_uploader(
            "Upload PDF(s) and click 'Process'",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("Process Documents ⚙️", key="process_button"):
            if uploaded_pdf_files:
                current_file_names = sorted([doc.name for doc in uploaded_pdf_files])
                if st.session_state.uploaded_file_names != current_file_names or not st.session_state.pdfs_processed:
                    with st.spinner("Processing documents... ⏳"):
                        try:
                            if not os.getenv("GOOGLE_API_KEY"):
                                st.error("GOOGLE_API_KEY not found. Please set it in your environment.")
                                return

                            st.write("Loading PDFs...")
                            documents = load_pdfs_with_pypdf(uploaded_pdf_files)
                            if not documents:
                                st.error("Could not extract documents. Check PDF files.")
                                return
                            
                            st.write(f"Splitting {len(documents)} document pages into chunks...")
                            document_chunks = get_document_chunks(documents)
                            if not document_chunks:
                                st.error("Could not split documents into chunks.")
                                return
                            st.write(f"Split into {len(document_chunks)} chunks.")

                            st.write("Creating vector store (HuggingFace Embeddings)...")
                            vector_store = get_vector_store(document_chunks)
                            if vector_store is None:
                                st.error("Failed to create vector store.")
                                return
                            st.write("Vector store created.")

                            st.write("Initializing conversational chain (Gemini)...")
                            st.session_state.conversation_chain = get_conversational_chain(vector_store)
                            st.session_state.pdfs_processed = True
                            st.session_state.uploaded_file_names = current_file_names
                            st.session_state.chat_history = []
                            st.success("✅ Documents processed! Ask your questions.")
                        except ValueError as ve:
                            st.error(f"Configuration Error: {ve}")
                        except Exception as e:
                            st.error(f"Processing Error: {e}")
                            print(traceback.format_exc())
                            st.session_state.pdfs_processed = False
                elif st.session_state.pdfs_processed:
                    st.info("Documents already processed. Ask away!")
            else:
                st.warning("Please upload PDF files first.")

    st.header("🤖 Doc-RAG")
    st.markdown("Upload PDFs, process, then ask questions. Retrieved sources will be shown.")

    # Display chat history
    if st.session_state.chat_history:
        for message_obj in st.session_state.chat_history:
            is_ai = isinstance(message_obj, AIMessage)
            message_type = "assistant" if is_ai else "user"
            avatar_icon = "🤖" if is_ai else "🧑‍💻"

            with st.chat_message(name=message_type, avatar=avatar_icon):
                st.markdown(message_obj.content)
                if is_ai and message_obj.additional_kwargs and 'source_documents' in message_obj.additional_kwargs:
                    source_docs = message_obj.additional_kwargs['source_documents']
                    if source_docs:
                        with st.expander("View Retrieved Sources", expanded=False):
                            for i, doc in enumerate(source_docs):
                                st.markdown(f"<div class='source-document'>", unsafe_allow_html=True)
                                st.markdown(f"**Source {i+1}**", unsafe_allow_html=True)
                                if doc.metadata:
                                    # Display original filename and page
                                    source_name = doc.metadata.get('source', 'Unknown Source')
                                    page_num = doc.metadata.get('page', 'N/A')
                                    st.markdown(f"<p class='metadata'><i>From: {source_name}, Page: {page_num + 1 if isinstance(page_num, int) else page_num}</i></p>", unsafe_allow_html=True)
                                st.markdown(f"<p>{doc.page_content}</p>", unsafe_allow_html=True)
                                st.markdown(f"</div>", unsafe_allow_html=True)
                                if i < len(source_docs) - 1:
                                    st.markdown("---") # Use st.divider() if preferred for thicker lines
    
    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        if not st.session_state.pdfs_processed or not st.session_state.conversation_chain:
            st.warning("Please upload and process PDF documents first.")
        else:
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            # Display user message immediately
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_question)

            with st.spinner("Gemini is thinking and retrieving sources... 🤔"):
                try:
                    ai_response_content, source_documents = handle_chat_interaction(
                        user_question,
                        st.session_state.conversation_chain,
                        st.session_state.chat_history # Pass the full history
                    )
                    
                    # Store AI message with its source documents
                    ai_message_with_sources = AIMessage(
                        content=ai_response_content,
                        additional_kwargs={'source_documents': source_documents}
                    )
                    st.session_state.chat_history.append(ai_message_with_sources)
                    
                    st.rerun() # Rerun to display the new AI message and its sources

                except Exception as e:
                    st.error(f"Error during chat: {e}")
                    print(traceback.format_exc())
                    # Optionally, add a placeholder AI error message to history
                    # st.session_state.chat_history.append(AIMessage(content=f"Sorry, an error occurred: {e}"))
                    # st.rerun()



if __name__ == '__main__':
    main()