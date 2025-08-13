FROM python:3.11-slim

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy deps first for better caching
COPY requirements.txt .

# Upgrade pip and install Python deps
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 # Install CPU-only torch from official index
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy app
COPY . .

# Expose (optional); Railway injects $PORT
EXPOSE 8000

# Run FastAPI; use Railway's $PORT if provided
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
