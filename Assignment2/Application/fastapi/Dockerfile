# Base image
FROM python:3.9-slim

# Install necessary dependencies for PostgreSQL and other system utilities
RUN apt-get update && apt-get install -y \
    libpq-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements.txt first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code to the working directory
COPY . .

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
