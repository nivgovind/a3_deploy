
# NOTE: NOT USED IN THIS PROJECT
# ALl endpoints moved to main.py
# NOT WORKING!!!!!


from fastapi import APIRouter, UploadFile, File, Form
from services.rag_service import RAGService
import tempfile
import os


# NOTE: NOT USED IN THIS PROJECT
# ALl endpoints moved to main.py
# NOT WORKING!!!!!

router = APIRouter()
rag_service = RAGService()

@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for file in files:
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        file_paths.append(temp_file_path)
    
    index = rag_service.process_files(file_paths)
    # Store the index (you might want to implement a proper storage solution)
    return {"message": "Files processed successfully"}

@router.post("/query")
async def query(query_text: str = Form(...)):
    # Retrieve the index (implement proper retrieval method)
    index = rag_service.get_index()
    response = rag_service.query(index, query_text)
    return {"response": response}