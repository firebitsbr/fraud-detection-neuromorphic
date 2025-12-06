# Neuromorphic Fraud Detection - Docker Container
#
# Description: Container Docker para execução do sistema de detecção de fraude
#              neuromórfica com todas as dependências necessárias.
#
# Author: Mauro Risonho de Paula Assumpção
# Created: December 5, 2025
# License: MIT License

FROM fraud-detection-base:latest

LABEL maintainer="Mauro Risonho de Paula Assumpção"
LABEL description="Neuromorphic Fraud Detection - Docker Container"
LABEL version="1.0.0"
LABEL license="MIT"

# Set working directory
WORKDIR /app

# Dependencies are already installed in base image

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
ENV PYTHONPATH=/app:/app/src
ENV JUPYTER_ENABLE_LAB=yes
ENV FLASK_APP=api/main.py

# Default command: Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
