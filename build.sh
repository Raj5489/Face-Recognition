#!/usr/bin/env bash
set -o errexit

echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    python3-dev \
    python3-pip \
    libopencv-dev \
    python3-opencv \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgif-dev \
    && rm -rf /var/lib/apt/lists/*

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing Python dependencies..."
pip install --no-cache-dir wheel setuptools cmake

# Install dlib with specific flags
export CFLAGS="-O2"
export CXXFLAGS="-O2"
pip install --no-cache-dir dlib --verbose

# Install the rest of the requirements
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
mkdir -p known_faces
chmod 777 known_faces

echo "Build completed successfully!"
