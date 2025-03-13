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
    wget \
    && rm -rf /var/lib/apt/lists/*

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing Python dependencies..."
pip install --no-cache-dir wheel setuptools

# Install cmake separately with specific version
pip install --no-cache-dir cmake==3.25.0

# Download and install pre-built dlib wheel
mkdir -p /tmp/wheels
cd /tmp/wheels
wget https://files.pythonhosted.org/packages/0e/ce/f8a3cff33ac03a8219768f0694c5d703c8e037e6aba2e865f9bae22ed63c/dlib-19.24.0-cp310-cp310-manylinux1_x86_64.whl
pip install dlib-19.24.0-cp310-cp310-manylinux1_x86_64.whl
cd -
rm -rf /tmp/wheels

# Install the rest of the requirements
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
mkdir -p known_faces
chmod 777 known_faces

echo "Build completed successfully!"
