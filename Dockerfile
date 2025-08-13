# =====================
# 1st Stage: Dependencies Build
# =====================
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Upgrade pip and install only dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# =====================
# 2nd Stage: Final Image
# =====================
FROM python:3.11-slim

WORKDIR /app

# Install only what's needed for runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local /usr/local

# Copy the rest of your source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]