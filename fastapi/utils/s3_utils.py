# utils/s3_utils.py

import boto3
from botocore.exceptions import ClientError
from config import fastapi_config
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3Client:
    @staticmethod
    def get_s3_client():
        return boto3.client(
            's3',
            aws_access_key_id=fastapi_config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=fastapi_config.AWS_SECRET_ACCESS_KEY,
            region_name=fastapi_config.AWS_REGION
        )

def list_buckets():
    s3 = S3Client.get_s3_client()
    response = s3.list_buckets()
    return response

def list_objects():
    s3 = S3Client.get_s3_client()
    response = s3.list_objects_v2(Bucket=fastapi_config.S3_BUCKET_NAME)
    return response

def upload_file(file_name, bucket_name=fastapi_config.S3_BUCKET_NAME):
    s3 = S3Client.get_s3_client()
    try:
        response = s3.upload_file(file_name, bucket_name, file_name)
    except ClientError as e:
        return e
    return response

def download_file(file_name, bucket_name):
    s3 = S3Client.get_s3_client()
    try:
        response = s3.download_file(bucket_name, file_name, file_name)
    except ClientError as e:
        return e
    return response

def check_connection():
    s3 = S3Client.get_s3_client()
    try:
        response = s3.list_buckets()
    except ClientError as e:
        return e
    return response

def get_document_details(document_name: str):
    s3_client = S3Client.get_s3_client()
    try:
        response = s3_client.head_object(Bucket=fastapi_config.S3_BUCKET_NAME, Key=document_name)
        details = {
            "ContentLength": response["ContentLength"],
            "ContentType": response["ContentType"],
            "LastModified": response["LastModified"].isoformat(),
            "ETag": response["ETag"]
        }
        return details
    except ClientError as e:
        raise e

def list_s3_documents():
    s3 = S3Client.get_s3_client()
    response = s3.list_objects_v2(Bucket=fastapi_config.S3_BUCKET_NAME)
    try:
        return [obj for obj in response.get('Contents', [])]
    except Exception as e:
        return []

def upload_file_to_s3(file_content, bucket_name, key):
    s3 = S3Client.get_s3_client()
    try:
        s3.put_object(Bucket=bucket_name, Key=key, Body=file_content)
        return True
    except ClientError as e:
        print(f"Error uploading file to S3: {e}")
        return False

def save_session_history_to_s3(filename, file_content):
    bucket_name = fastapi_config.S3_BUCKET_NAME
    key = f'staging/research_doc/{filename}'
    return upload_file_to_s3(file_content, bucket_name, key)


# utils/s3_utils.py

import boto3
from botocore.exceptions import ClientError
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

# Initialize S3 client
s3_client = S3Client.get_s3_client()

S3_BUCKET_NAME = fastapi_config.S3_BUCKET_NAME
RESEARCH_NOTES_PREFIX = "research_notes/"  # S3 folder prefix for research notes

def save_research_note_to_s3(document_id: str, research_note: str) -> bool:
    """
    Saves a research note as a text file in AWS S3 under the research_notes/{document_id}/ directory.
    """
    try:
        # Generate a unique filename, e.g., timestamp-based or UUID
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"research_note_{timestamp}.txt"
        s3_key = f"{RESEARCH_NOTES_PREFIX}{document_id}/{filename}"
        
        # Upload the research note as a text file
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=research_note.encode('utf-8'),
            ContentType='text/plain'
        )
        logger.info(f"Research note saved to S3 at {s3_key}.")
        return True
    except ClientError as e:
        logger.error(f"Failed to save research note to S3: {e}")
        return False

def get_research_notes_from_s3(document_id: str) -> List[str]:
    """
    Retrieves all research notes for a given document ID from AWS S3.
    Returns a list of research note contents.
    """
    try:
        s3_key_prefix = f"{RESEARCH_NOTES_PREFIX}{document_id}/"
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=s3_key_prefix
        )
        
        if 'Contents' not in response:
            logger.info(f"No research notes found for document ID: {document_id}.")
            return []
        
        research_notes = []
        for obj in response['Contents']:
            key = obj['Key']
            # Skip the directory itself
            if key.endswith('/'):
                continue
            # Fetch the object
            obj_response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
            content = obj_response['Body'].read().decode('utf-8')
            research_notes.append(content)
        
        logger.info(f"Fetched {len(research_notes)} research notes from S3 for document ID: {document_id}.")
        return research_notes
    except ClientError as e:
        logger.error(f"Failed to retrieve research notes from S3: {e}")
        return []
