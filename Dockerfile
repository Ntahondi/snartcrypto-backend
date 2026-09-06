# Use official lightweight Python 3.12 Linux base image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered UTF-8 logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TZ=UTC

# Install essential Linux system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project source code into container
COPY . .

# Install local SnailGuard Python SDK if present
RUN if [ -f "./snailguard_python_SDK/setup.py" ]; then pip install --no-cache-dir -e ./snailguard_python_SDK; fi

# Create persistent directory folders inside container
RUN mkdir -p data models positions signal_history logs

# Expose FastAPI Uvicorn Port
EXPOSE 8000

# Health check endpoint verification
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "main.py"]