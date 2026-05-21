FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dulu (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua source code
COPY . .

# HF Spaces wajib pakai port 7860
EXPOSE 7860

# Jalankan uvicorn di port 7860
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "7860"]