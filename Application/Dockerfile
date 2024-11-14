FROM apache/airflow:2.10.2
USER root
RUN apt-get update && apt-get install -y git git-lfs
RUN git lfs install
RUN apt-get update && apt-get install -y git
RUN mkdir -p /tmp/gaia \
    && chown -R airflow /tmp/gaia \
    && chmod -R 775 /tmp/gaia

USER airflow
RUN pip install --no-cache-dir \
    boto3 \
    gitpython \
    python-dotenv \
    requests \
    azure-ai-formrecognizer \
    PyPDF2

RUN pip install apache-airflow-providers-amazon
