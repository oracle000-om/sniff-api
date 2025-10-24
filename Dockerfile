FROM python:3.11-slim

WORKDIR /app

# Install ALL system dependencies for OpenCV and ML
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    wget \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgfortran5 \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p data/images data/backups

# Expose port
EXPOSE 8000

# Start app
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}