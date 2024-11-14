
# NOTE: NOT USED IN THIS PROJECT
# ALl endpoints moved to main.py
# NOT WORKING!!!!!



# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.vector_stores.milvus import MilvusVectorStore
# from utils.document_processors import process_file
# from utils import get_nim_llm, get_nim_embedding

class RAGService:
    def __init__(self):
#         self.vector_store = MilvusVectorStore(
#             host="127.0.0.1",
#             port=19530,
#             dim=1024,
#             collection_name="multimodal_rag"
#         )
#         self.llm = get_nim_llm()
#         self.embed_model = get_nim_embedding()
        self.index = None

#     def process_files(self, file_paths):
#         documents = []
#         for file_path in file_paths:
#             doc = process_file(file_path)
#             if doc:
#                 documents.append(doc)
        
#         self.index = VectorStoreIndex.from_documents(
#             documents,
#             vector_store=self.vector_store,
#             embed_model=self.embed_model
#         )
#         return self.index

#     def query(self, query_text):
#         if not self.index:
#             raise ValueError("No index available. Please process files first.")
#         query_engine = self.index.as_query_engine(
#             llm=self.llm,
#             similarity_top_k=5
#         )
#         response = query_engine.query(query_text)
#         return response.response

    def get_index(self):
        return self.index
