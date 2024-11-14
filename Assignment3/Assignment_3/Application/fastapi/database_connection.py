# database_connection.py

import os
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
from config import fastapi_config

load_dotenv()

DATABASE_URL = fastapi_config.DATABASE_URL

# Retry configuration
max_retries = 5
retry_delay = 5  # in seconds

# SQLAlchemy database setup
engine = None
for attempt in range(max_retries):
    try:
        engine = create_engine(DATABASE_URL)
        connection = engine.connect()  # Test the connection
        connection.close()
        print("Successfully connected to the database.")
        break
    except OperationalError:
        print(f"Database connection failed. Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
else:
    raise Exception(f"Failed to connect to the database after {max_retries} attempts.")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency for getting the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
