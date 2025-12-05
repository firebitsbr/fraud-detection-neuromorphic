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

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY ../src /app/src
COPY ../notebooks /app/notebooks

# Create directories for models and data
RUN mkdir -p /app/models /app/data

# Expose JupyterLab port
EXPOSE 8888

# Set environment variables
ENV PYTHONPATH=/app
ENV JUPYTER_ENABLE_LAB=yes

# Default command: Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
