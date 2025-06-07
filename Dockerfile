FROM python:3.9-slim

WORKDIR /app

# Install minimal dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install (using minimal versions)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories
RUN mkdir -p models visualizations data logs

# Environment
ENV PYTHONPATH=/app
ENV FLASK_ENV=production

EXPOSE 5000

# Simple health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "src/app.py"]