# utils/initialization.py
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from datetime import datetime
from typing import List, Dict, Any, Optional
from config import fastapi_config

ZILLIZ_CLOUD_URI=fastapi_config.ZILLIZ_CLOUD_URI
ZILLIZ_CLOUD_API_KEY=fastapi_config.ZILLIZ_CLOUD_API_KEY

def initialize_settings():
    Settings.embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", truncate="END")
    Settings.llm = NVIDIA(model="meta/llama-3.1-70b-instruct")
    Settings.text_splitter = SentenceSplitter(chunk_size=600)

def create_index(documents):
    # vector_store = MilvusVectorStore(uri="././milvus_demo.db", dim=1024, overwrite=True)
    vector_store = MilvusVectorStore(uri=ZILLIZ_CLOUD_URI, token=ZILLIZ_CLOUD_API_KEY, dim=1024, overwrite=True, collection_name="bigdata", drop_old=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

def get_embeddings_for_document(document_id):
    vector_store = MilvusVectorStore(uri=ZILLIZ_CLOUD_URI, token=ZILLIZ_CLOUD_API_KEY, dim=1024, overwrite=False)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    query_results = vector_store.query({"doc_id": document_id})
    
    return query_results

# def create_index(documents):
#     # Extract document ID for naming purposes
#     document_id = documents[0].metadata.get("filter_doc_id")
#     collection_nm = f"bigdata_{document_id}"

#     # Initialize the vector store
#     vector_store = MilvusVectorStore(
#         uri=ZILLIZ_CLOUD_URI,
#         token=ZILLIZ_CLOUD_API_KEY,
#         dim=1024,
#         overwrite=False,
#         collection_name=collection_nm
#     )
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)

#     try:
#         # Attempt to load existing index if it exists
#         index = VectorStoreIndex(storage_context=storage_context)
#         print(f"Loaded existing index for collection: {collection_nm}")
#         return index
#     except Exception as e:
#         print(f"No existing index found for {collection_nm}. Creating a new index. Error: {e}")
#         # If loading fails, create a new index with the provided documents
#         index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
#         print(f"Created new index for collection: {collection_nm}")
#         return index


# def create_index(documents):
#     # Extract document ID for naming purposes
#     document_id = documents[0].metadata.get("filter_doc_id")
#     collection_nm = f"bigdata_{document_id}"

#     # Initialize the vector store
#     vector_store = MilvusVectorStore(
#         uri=ZILLIZ_CLOUD_URI,
#         token=ZILLIZ_CLOUD_API_KEY,
#         dim=1024,
#         overwrite=False,
#         collection_name=collection_nm
#     )
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)

#     try:
#         # Use a dummy vector for querying, assuming 1024 dimensions
#         dummy_query_vector = [0.0] * 1024

#         # Attempt to query the collection with the dummy vector and a filter
#         existing_entries = vector_store.query(
#             query=dummy_query_vector,
#             top_k=1,
#             filter={"filter_doc_id": document_id}
#         )

#         # Debug print to inspect `existing_entries`
#         print(f"Query result for existing entries: {existing_entries}")

#         if existing_entries and len(existing_entries) > 0:
#             print(f"Existing index found for document ID: {document_id}")
#             # Load and return the existing index
#             return VectorStoreIndex(storage_context=storage_context)
#         else:
#             print(f"No existing entries found for document ID: {document_id}, creating new index.")
#             # If no entries exist, create a new index with the provided documents
#             return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

#     except Exception as e:
#         print(f"Error while checking for existing entries: {e}")
#         raise HTTPException(status_code=500, detail="Failed to create or load index.")
