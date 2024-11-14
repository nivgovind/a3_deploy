#fastapi/servies/document_service.py
from pydantic import BaseModel
from utils.snowflake_client import SnowflakeClient

# Ensure DocumentSelection and DocumentInfo are defined or imported correctly
class DocumentSelection(BaseModel):
    document_id: str

class DocumentInfo(BaseModel):
    document_id: str
    title: str
    brief_summary: str
    image_link: str
    pdf_link: str

async def get_document_by_selection(selection: DocumentSelection) -> DocumentInfo:
    snowflake_client = SnowflakeClient()
    document = snowflake_client.fetch_document_by_id(selection.document_id)
    if not document:
        raise ValueError("Document not found")
    return DocumentInfo(**document)
