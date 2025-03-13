#!/bin/bash
set -e

echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    python3-opencv \
    python3-dev \
    python3-pip \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

echo "Upgrading pip..."
python -m pip install --no-cache-dir --upgrade pip

echo "Installing Python dependencies..."
pip install --no-cache-dir wheel setuptools cmake
pip install --no-cache-dir dlib==19.24.2
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
mkdir -p known_faces
chmod 777 known_faces

echo "Build completed successfully!"
