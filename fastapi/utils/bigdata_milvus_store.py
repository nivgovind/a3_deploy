# bigdata_milvus_store.py

import json
import sqlite3
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
import logging
import os

logger = logging.getLogger(__name__)

class MilvusVectorStore:
    def __init__(self, uri: str, token: str, dim: int, overwrite: bool = False, metadata_db: str = "metadata.db"):
        """
        Initializes the Milvus Vector Store connected to Zilliz Cloud.

        Args:
            uri (str): The URI of the Milvus server (e.g., "your-cluster-name.zillizcloud.com:19530").
            token (str): The authentication token for Zilliz Cloud.
            dim (int): The dimensionality of the vectors.
            overwrite (bool): Whether to overwrite existing collections.
            metadata_db (str): Path to the SQLite metadata database.
        """
        self.uri = uri
        self.token = token
        self.dim = dim
        self.overwrite = overwrite
        self.metadata_db = metadata_db

        # Establish connection to Milvus on Zilliz Cloud
        try:
            connections.connect(
                alias="default",
                uri=self.uri,
                token=self.token,
                secure=True 
            )
            logger.info(f"Connected to Milvus at {self.uri} on Zilliz Cloud.")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus at {self.uri}: {e}")
            raise e

        # Initialize metadata database
        self.initialize_metadata_db()

    def initialize_metadata_db(self):
        """
        Initializes the metadata database with necessary tables.
        """
        try:
            self.metadata_conn = sqlite3.connect(self.metadata_db)
            self.metadata_cursor = self.metadata_conn.cursor()
            self.metadata_cursor.execute("""
                CREATE TABLE IF NOT EXISTS collection_metadata (
                    collection_name TEXT PRIMARY KEY,
                    metadata TEXT
                )
            """)
            self.metadata_conn.commit()
            logger.info("Metadata database initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing metadata database: {e}")
            raise e

    def index_exists(self, collection_name: str) -> bool:
        """
        Checks if a collection exists in Milvus.

        Args:
            collection_name (str): The name of the collection/document.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        try:
            exists = Collection.exists(collection_name)
            logger.debug(f"Collection '{collection_name}' exists: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking if collection exists: {e}")
            raise e

    def create_collection(self, collection_name: str) -> Collection:
        """
        Creates a new collection in Milvus.

        Args:
            collection_name (str): The name of the collection to create.

        Returns:
            Collection: The created Milvus collection.
        """
        try:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]

            schema = CollectionSchema(fields=fields, description="Vector store collection")

            collection = Collection(name=collection_name, schema=schema)
            logger.info(f"Collection '{collection_name}' created successfully.")
            return collection
        except Exception as e:
            logger.error(f"Error creating collection '{collection_name}': {e}")
            raise e

    def load_collection(self, collection_name: str) -> Collection:
        """
        Loads an existing collection from Milvus.

        Args:
            collection_name (str): The name of the collection to load.

        Returns:
            Collection: The loaded Milvus collection.
        """
        try:
            collection = Collection(collection_name)
            logger.info(f"Collection '{collection_name}' loaded successfully.")
            return collection
        except Exception as e:
            logger.error(f"Error loading collection '{collection_name}': {e}")
            raise e

    def update_metadata(self, collection_name: str, additional_metadata: dict):
        """
        Updates the metadata of a collection by storing it in the metadata database.

        Args:
            collection_name (str): The name of the collection.
            additional_metadata (dict): A dictionary of metadata fields to update.
        """
        try:
            # Fetch existing metadata
            self.metadata_cursor.execute("""
                SELECT metadata FROM collection_metadata WHERE collection_name=?
            """, (collection_name,))
            row = self.metadata_cursor.fetchone()
            if row:
                existing_metadata = json.loads(row[0])
            else:
                existing_metadata = {}

            # Update with additional metadata
            existing_metadata.update(additional_metadata)

            if row:
                # Update existing record
                self.metadata_cursor.execute("""
                    UPDATE collection_metadata SET metadata=? WHERE collection_name=?
                """, (json.dumps(existing_metadata), collection_name))
            else:
                # Insert new record
                self.metadata_cursor.execute("""
                    INSERT INTO collection_metadata (collection_name, metadata) VALUES (?, ?)
                """, (collection_name, json.dumps(existing_metadata)))

            self.metadata_conn.commit()
            logger.info(f"Metadata for collection '{collection_name}' updated to: {existing_metadata}")
        except Exception as e:
            logger.error(f"Error updating metadata for collection '{collection_name}': {e}")
            raise e

    def get_metadata(self, collection_name: str) -> dict:
        """
        Retrieves metadata for a given collection.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            dict: The metadata dictionary.
        """
        try:
            self.metadata_cursor.execute("""
                SELECT metadata FROM collection_metadata WHERE collection_name=?
            """, (collection_name,))
            row = self.metadata_cursor.fetchone()
            if row:
                return json.loads(row[0])
            else:
                return {}
        except Exception as e:
            logger.error(f"Error retrieving metadata for collection '{collection_name}': {e}")
            raise e

    def __del__(self):
        """
        Ensures that the metadata database connection is closed upon deletion of the instance.
        """
        try:
            self.metadata_conn.close()
            logger.info("Metadata database connection closed.")
        except Exception as e:
            logger.error(f"Error closing metadata database: {e}")


