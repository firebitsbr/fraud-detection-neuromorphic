# Neuromorphic Fraud Detection - Production Dockerfile
# Base image: Ubuntu 24.04.3 LTS Server Minimal
#
# Multi-stage build for optimized production image
#
# Author: Mauro Risonho de Paula Assumpção
# Email: mauro.risonho@gmail.com
# LinkedIn: linkedin.com/in/maurorisonho
# GitHub: github.com/maurorisonho
# Date: December 2025
# License: MIT

# ============================================================================
# Stage 1: Builder - Compile dependencies
# ============================================================================
FROM ubuntu:24.04 AS builder

LABEL maintainer="Mauro Risonho <mauro.risonho@gmail.com>"
LABEL description="Builder stage for neuromorphic fraud detection"
LABEL version="1.0.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build essentials and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
WORKDIR /build
COPY requirements.txt requirements-ci.txt requirements-edge.txt ./

# Install Python dependencies in layers (most stable first)
# Note: scipy 1.14.1 requires numpy<2.3
RUN pip install --no-cache-dir \
    numpy==2.2.1 \
    scipy==1.14.1 \
    pandas==2.2.3

RUN pip install --no-cache-dir \
    matplotlib==3.10.7 \
    seaborn==0.13.2 \
    plotly==5.24.1

RUN pip install --no-cache-dir \
    scikit-learn==1.5.2 \
    Brian2==2.10.1

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    snntorch==0.9.1

RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    uvicorn[standard]==0.32.1 \
    pydantic==2.10.4

RUN pip install --no-cache-dir \
    kafka-python==2.0.2 \
    redis==5.2.1 \
    psutil==7.1.3

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt || true

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM ubuntu:24.04 AS runtime

LABEL maintainer="Mauro Risonho <mauro.risonho@gmail.com>"
LABEL description="Neuromorphic Fraud Detection - Production Runtime"
LABEL version="1.0.0"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install only runtime dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    libopenblas0 \
    libgomp1 \
    libhdf5-103 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app directory and user
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 -m -s /bin/bash appuser

WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser hardware/ ./hardware/
COPY --chown=appuser:appuser scaling/ ./scaling/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser examples/ ./examples/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python3", "-m", "api.main"]

# Expose port
EXPOSE 8000

# ============================================================================
# Metadata
# ============================================================================
LABEL org.opencontainers.image.source="https://github.com/maurorisonho/fraud-detection-neuromorphic"
LABEL org.opencontainers.image.description="Neuromorphic Fraud Detection System using SNNs"
LABEL org.opencontainers.image.licenses="MIT"
