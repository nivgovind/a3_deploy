# main.py

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv
import os
import logging

from config import fastapi_config
from database_connection import engine, SessionLocal, Base, get_db
from models import User
from utils.s3_utils import (
    list_buckets,
    list_s3_documents,
    upload_file_to_s3,
    save_session_history_to_s3,
    save_research_note_to_s3,
    get_research_notes_from_s3,
)
from utils.snowflake_client import SnowflakeClient
from utils.initialization import initialize_settings, create_index, get_embeddings_for_document
from utils.document_processors import (
    load_document_by_id,
    generate_summary,
    extract_text_from_pdf,
)
from llama_index.core import Document
from llama_index.llms.nvidia import NVIDIA

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the LLM model once
mixtral = NVIDIA(model_name="meta/llama-3.1-70b-instruct")

# Initialize settings for NVIDIA API
initialize_settings()

# FastAPI application instance
app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Streamlit app's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create all database tables
Base.metadata.create_all(bind=engine)

# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = fastapi_config.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Dictionary to store indices per document
INDICES: Dict[str, Any] = {}

# Data Models
class InitializeEmbeddingsRequest(BaseModel):
    document_id: str

class GenerateSummaryRequest(BaseModel):
    document_id: str

class QueryRequest(BaseModel):
    query: str
    document_id: str

class SaveSessionRequest(BaseModel):
    document_id: str
    session_history: List[Dict[str, Any]]   # List of messages with possible 'satisfied' field

class GenerateReportRequest(BaseModel):
    document_id: str
    query: str

class GenerateReportResponse(BaseModel):
    report: str

class SaveResearchNoteRequest(BaseModel):
    document_id: str
    research_note: str

class SaveEntireResearchNoteRequest(BaseModel):
    document_id: str
    research_note: str

class GetResearchNotesResponse(BaseModel):
    research_notes: List[str]

class GetDocumentInfoResponse(BaseModel):
    DOC_ID: str
    TITLE: str
    SUMMARY: Optional[str]
    IMAGELINK: Optional[str]
    PDFLINK: Optional[str]
    text: Optional[str]

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

# Authentication Utilities
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)  # Default expiration
    to_encode.update({"exp": int(expire.timestamp())})  # Convert to UNIX timestamp
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# User Registration Endpoint
@app.post("/register", response_model=UserCreate)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == user_data.username).first()
    if user:
        raise HTTPException(status_code=400, detail="User already exists")
    hashed_password = get_password_hash(user_data.password)
    new_user = User(username=user_data.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# User Login/Token Generation Endpoint
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

# Protected Endpoint Example
@app.get("/protected-endpoint")
def protected_endpoint(current_user: User = Depends(get_current_user)):
    return {"message": f"Hello, {current_user.username}. You are authenticated!"}

# Existing FastAPI Endpoints (Protected)
@app.get("/list_documents_info", response_model=List[GetDocumentInfoResponse])
def list_documents_info(current_user: User = Depends(get_current_user)):
    fetch_and_cache_documents()
    return DOCUMENTS_CACHE

@app.post("/initialize_embeddings")
def initialize_embeddings_endpoint(request: InitializeEmbeddingsRequest, current_user: User = Depends(get_current_user)):
    try:
        logger.info(f"Loading document with ID: {request.document_id}")
        # Load the document content based on document_id
        document = load_document_by_id(request.document_id)

        if not document:
            logger.error(f"Document with ID {request.document_id} not found.")
            raise HTTPException(status_code=404, detail="Document not found.")
        print("document-id : ", document.metadata.get("filter_doc_id"))
        print("document_id : ", request.document_id)
        # Create and store index for the document
        print("embedding1 : ", document)
        index = create_index([document])
        INDICES[request.document_id] = index
        print("embedding2 : ", INDICES)
        print("embedding3 : ", index)

        logger.info("Embeddings initialized successfully.")
        return {"message": "Embeddings initialized successfully"}
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/generate_summary")
def generate_summary_endpoint(request: GenerateSummaryRequest, current_user: User = Depends(get_current_user)):
    try:
        logger.info(f"Request received for document ID: {request.document_id}")
        
        document = load_document_by_id(request.document_id)
        logger.debug(f"Document loaded: {document}")
        
        if not document:
            logger.error("Document not found")
            raise ValueError("Document not found")

        summary = generate_summary(document.text)
        logger.debug(f"Summary generated: {summary}")
        
        return {"summary": summary}
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_index_endpoint(request: QueryRequest, current_user: User = Depends(get_current_user)):
    document_id = request.document_id
    user_query = request.query
    index = INDICES.get(document_id)

    if not index:
        logger.error("Index not initialized for this document")
        raise HTTPException(status_code=400, detail="Index not initialized for this document")

    try:
        # Define the formatting instruction
        formatting_instruction = (
            "Format your response as detailed research notes. "
            "Ensure that the notes include links to relevant graphs, tables, and specific page numbers where applicable.\n\n"
        )

        # Combine the instruction with the user's query
        formatted_query = formatting_instruction + user_query

        # Use the index to perform the query with the formatted query
        response = index.as_query_engine(similarity_top_k=20).query(formatted_query)
        full_response = response.response

        return JSONResponse(content={"response": full_response})
    except Exception as e:
        logger.error(f"Exception during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_session_history")
def save_session_history_endpoint(request: SaveSessionRequest, current_user: User = Depends(get_current_user)):
    try:
        logger.info(f"Saving session history for document ID: {request.document_id}")
        # Convert session history to text, including satisfaction
        session_text = ""
        for message in request.session_history:
            role = message["role"].capitalize()
            content = message["content"]
            session_text += f"{role}: {content}\n"
            if role == "Assistant" and "satisfied" in message:
                satisfaction = "Yes" if message["satisfied"] else "No"
                session_text += f"User Satisfaction: {satisfaction}\n"

        # Save the session history as a text file
        text_content = session_text.encode('utf-8')  # Convert to bytes

        # Fetch document details
        snowflake_client = SnowflakeClient()
        document = snowflake_client.fetch_document_content_by_id(request.document_id)
        snowflake_client.close_connection()

        if not document:
            logger.error("Document not found")
            raise HTTPException(status_code=404, detail="Document not found")

        title = document["TITLE"]
        # Replace spaces with underscores and remove problematic characters for filenames
        safe_title = "".join([c if c.isalnum() or c in (" ", "_") else "_" for c in title]).replace(" ", "_")
        filename = f"{safe_title}_session_history.txt"

        # Save text to S3
        if save_session_history_to_s3(filename, text_content):
            logger.info("Session history saved successfully.")
            return {"message": "Session history saved successfully"}
        else:
            logger.error("Failed to save session history to S3.")
            raise HTTPException(status_code=500, detail="Failed to save session history.")
    except Exception as e:
        logger.error(f"Exception while saving session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_report", response_model=GenerateReportResponse)
def generate_report_endpoint(request: GenerateReportRequest, current_user: User = Depends(get_current_user)):
    try:
        logger.info(f"Generating report for document ID: {request.document_id}")
        
        # Load the document
        document = load_document_by_id(request.document_id)
        if not document:
            logger.error(f"Document with ID {request.document_id} not found.")
            raise HTTPException(status_code=404, detail="Document not found.")

        # Generate the report
        report = generate_report(document, request.query)
        if not report:
            logger.error("Report generation failed.")
            raise HTTPException(status_code=500, detail="Report generation failed.")
        
        return {"report": report}
    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise he
    except Exception as e:
        logger.exception("An error occurred during report generation.")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/save_research_note")
def save_research_note_endpoint(request: SaveResearchNoteRequest, current_user: User = Depends(get_current_user)):
    try:
        logger.info(f"Saving research note for document ID: {request.document_id}")
        
        # Save the research note to S3
        success = save_research_note_to_s3(request.document_id, request.research_note)
        
        if success:
            logger.info("Research note saved successfully to S3.")
            return {"message": "Research note saved successfully."}
        else:
            logger.error("Failed to save research note to S3.")
            raise HTTPException(status_code=500, detail="Failed to save research note.")
    except Exception as e:
        logger.exception("An error occurred while saving the research note.")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/get_research_notes", response_model=GetResearchNotesResponse)
def get_research_notes_endpoint(document_id: str, current_user: User = Depends(get_current_user)):
    try:
        logger.info(f"Fetching research notes for document ID: {document_id}")
        
        # Retrieve research notes from S3
        notes = get_research_notes_from_s3(document_id)
        
        return {"research_notes": notes}
    except Exception as e:
        logger.exception("An error occurred while fetching research notes.")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/save_entire_research_note")
def save_entire_research_note_endpoint(request: SaveEntireResearchNoteRequest, current_user: User = Depends(get_current_user)):
    try:
        logger.info(f"Saving entire research note for document ID: {request.document_id}")
        
        # Save the research note to S3
        success = save_research_note_to_s3(request.document_id, request.research_note)
        
        if success:
            logger.info("Entire research note saved successfully to S3.")
            return {"message": "Entire research note saved successfully."}
        else:
            logger.error("Failed to save entire research note to S3.")
            raise HTTPException(status_code=500, detail="Failed to save entire research note.")
    except Exception as e:
        logger.exception("An error occurred while saving the entire research note.")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Fetch and Cache Documents
DOCUMENTS_CACHE = []

def fetch_and_cache_documents():
    global DOCUMENTS_CACHE
    if not DOCUMENTS_CACHE:
        try:
            snowflake_client = SnowflakeClient()
            df = snowflake_client.fetch_document_info()
            if df.empty:
                logger.warning("No documents found in the PUBLICATIONS table.")
            DOCUMENTS_CACHE = df.to_dict(orient="records")
            snowflake_client.close_connection()
        except Exception as e:
            logger.error(f"Error fetching documents: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch documents.")

# Report Generation Utility
def generate_report(document: Document, query: str) -> str:
    document_id = document
    user_query = query
    index = INDICES.get(document_id)

    if not index:
        logger.error("Index not initialized for this document")
        raise HTTPException(status_code=400, detail="Index not initialized for this document")

    try:
        # Define the formatting instruction
        formatting_instruction = (
            "Format your response as detailed research notes. "
            "Ensure that the notes include links to relevant graphs, tables, and specific page numbers where applicable.\n\n"
        )

        # Combine the instruction with the user's query
        formatted_query = formatting_instruction + user_query

        # Use the index to perform the query with the formatted query
        response = index.as_query_engine(similarity_top_k=20).query(formatted_query)
        full_response = response.response

        return full_response
    except Exception as e:
        logger.error(f"Exception during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
