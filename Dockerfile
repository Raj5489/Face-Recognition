FROM python:3.10.11-slim-buster

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    python3-opencv \
    python3-dev \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir wheel setuptools cmake && \
    pip install --no-cache-dir dlib==19.24.2 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for face images
RUN mkdir -p known_faces && chmod 777 known_faces

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Expose the port
EXPOSE 5000

# Start the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app --timeout 120 
