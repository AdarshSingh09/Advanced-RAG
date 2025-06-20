**Built an Advanced RAG-based QA System with Multi-Stage Retrieval and Reranking**

- Developed a Retrieval-Augmented Generation (RAG) QA web app using Streamlit, integrating `Gemini Flash 2.0` as the LLM for contextual, accurate answer generation.
- Employed LangChain's recursive text splitter for efficient document chunking, `FAISS` for fast vector similarity search, and `all-mpnet-base-v2` for high-quality sentence embeddings.
- Enhanced retrieval precision with Multi-Query Retriever (using MMR search) and Cross-Encoder Reranking via `BAAI/bge-reranker-base`, applying a ContextualCompressionRetriever to prioritize the top-k relevant documents.
