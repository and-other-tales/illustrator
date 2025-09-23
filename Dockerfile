# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY pyproject.toml ./

# Install Python dependencies including web dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[web,dev]"

# Copy application code
COPY . .

# Install the application in editable mode
RUN pip install -e .

# Copy and set permissions for start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Create directories for application data
RUN mkdir -p /app/saved_manuscripts /app/illustrator_output

# Set environment variables for Cloud Run
ENV HOST=0.0.0.0
ENV PORT=8080
ENV PYTHONPATH=/app/src

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Use start script as entrypoint
CMD ["/app/start.sh"]