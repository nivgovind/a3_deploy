# utils/snowflake_client.py

import snowflake.connector
import pandas as pd
from config import fastapi_config
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SnowflakeClient:
    def __init__(self):
        try:
            self.conn = snowflake.connector.connect(
                account=fastapi_config.SNOWFLAKE_ACCOUNT,
                user=fastapi_config.SNOWFLAKE_USER,
                password=fastapi_config.SNOWFLAKE_PASSWORD,
                warehouse=fastapi_config.SNOWFLAKE_WAREHOUSE,
                database=fastapi_config.SNOWFLAKE_DATABASE,
                schema=fastapi_config.SNOWFLAKE_SCHEMA,
                role=fastapi_config.SNOWFLAKE_ROLE
            )
            self._document_info_cache = None  # Initialize cache
            logger.info("Successfully connected to Snowflake.")
        except snowflake.connector.errors.Error as e:
            logger.error(f"Error connecting to Snowflake: {e}")
            raise

    def fetch_document_info(self) -> pd.DataFrame:
        if self._document_info_cache is not None:
            logger.debug("Using cached document info.")
            return self._document_info_cache

        query = """
        SELECT DOC_ID, TITLE, SUMMARY, IMAGELINK, PDFLINK
        FROM PUBLICATIONS
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)
                self._document_info_cache = df  # Cache the DataFrame
                logger.info(f"Fetched document info: {df.shape[0]} records.")
                return df
        except snowflake.connector.errors.ProgrammingError as e:
            logger.error(f"Error executing query: {e}")
            raise

    # def fetch_document_content_by_id(self, document_id: int) -> Optional[Dict[str, Any]]:
    #     df = self.fetch_document_info()
    #     print(f'Snowflake {df["DOC_ID"].head()}')
    #     logger.debug(f"DataFrame fetched: {df.shape[0]} records.")
    #     document = df[df["DOC_ID"] == document_id]
    #     print(f'Snowflake2 {document}')
    #     print(f'Snowflake3 {document_id}')
    #     if not document.empty:
    #         logger.info(f"Document with DOC_ID {document_id} found.")
    #         return {
    #             "SUMMARY": document.iloc[0]["SUMMARY"],
    #             "TITLE": document.iloc[0]["TITLE"],
    #             "PDFLINK": document.iloc[0]["PDFLINK"],
    #         }
    #     else:
    #         logger.warning(f"Document with DOC_ID {document_id} not found.")
            return None
        
    def fetch_document_content_by_id(self, document_id: int) -> Optional[Dict[str, Any]]:
        query = """
        SELECT DOC_ID, TITLE, SUMMARY, IMAGELINK, PDFLINK
        FROM PUBLICATIONS
        WHERE DOC_ID = %s
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, (document_id,))
                result = cursor.fetchone()
                if result:
                    logger.info(f"Document with DOC_ID {document_id} found.")
                    return {
                        "SUMMARY": result[2],
                        "TITLE": result[1],
                        "PDFLINK": result[4],
                    }
                else:
                    logger.warning(f"Document with DOC_ID {document_id} not found.")
                    return None
        except snowflake.connector.errors.ProgrammingError as e:
            logger.error(f"Error executing query: {e}")
            raise

    def close_connection(self):
        if self.conn:
            try:
                self.conn.close()
                logger.info("Snowflake connection closed.")
            except snowflake.connector.errors.Error as e:
                logger.error(f"Error closing Snowflake connection: {e}")
