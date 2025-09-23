#!/bin/bash

# Start script for Google Cloud Run deployment
# This script sets up the environment and starts the illustrator web application

echo "Starting Manuscript Illustrator..."

# Set the host and port for Cloud Run
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8080}

# Ensure we're in the correct directory
cd /app

# Start the application
echo "Starting illustrator on ${HOST}:${PORT}..."
exec illustrator start --host ${HOST} --port ${PORT} --no-open-browser