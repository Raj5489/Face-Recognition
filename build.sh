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
pip install --no-cache-dir wheel setuptools

# Install cmake separately with specific version
pip install --no-cache-dir cmake==3.25.0

# Configure environment for minimal memory usage during compilation
export CFLAGS="-O2 -g0"
export CXXFLAGS="-O2 -g0"
export MAKEFLAGS="-j1"
# Limit memory usage during compilation
export CARGO_NET_GIT_FETCH_WITH_CLI=true
export RUSTFLAGS="-C codegen-units=1"

# Install dlib with minimal build settings
pip install --no-cache-dir --no-deps dlib==19.24.0

# Install face_recognition after dlib
pip install --no-cache-dir face_recognition==1.3.0

# Install the rest of the requirements
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
mkdir -p known_faces
chmod 777 known_faces

echo "Build completed successfully!"
