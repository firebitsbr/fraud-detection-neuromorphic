# Neuromorphic Fraud Detection - Docker Container
#
# Description: Container Docker para execução do sistema de detecção de fraude
#              neuromórfica com todas as dependências necessárias.
#
# Author: Mauro Risonho de Paula Assumpção
# Created: December 5, 2025
# License: MIT License

FROM python:3.10-slim

LABEL maintainer="Mauro Risonho de Paula Assumpção"
LABEL description="Neuromorphic Fraud Detection - Docker Container"
LABEL version="1.0.0"
LABEL license="MIT"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements - use CI version or full version
COPY docker/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY src /app/src
COPY api /app/api
COPY notebooks /app/notebooks
COPY hardware /app/hardware
COPY scaling /app/scaling
COPY tests /app/tests

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs

# Expose ports
EXPOSE 8888 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV JUPYTER_ENABLE_LAB=yes
ENV FLASK_APP=api/app.py

# Default command: Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
