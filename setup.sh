#!/bin/bash

# Setup Script for llama-cpp-rs
# This script installs all necessary dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Setup Script for llama-cpp-rs${NC}"
echo "===================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root. This is not recommended for development."
fi

# Update package list
print_status "Updating package list..."
sudo apt update

# Install basic build tools
print_status "Installing build essentials..."
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    git \
    curl \
    wget

# Install Rust if not present
if ! command -v rustc &> /dev/null; then
    print_status "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
else
    print_status "Rust is already installed: $(rustc --version)"
fi

# Install additional dependencies for llama.cpp
print_status "Installing llama.cpp dependencies..."
sudo apt install -y \
    libomp-dev \
    libopenblas-dev \
    liblapack-dev

# Check for CUDA
print_status "Checking for CUDA installation..."
if [ -d "/usr/local/cuda" ] || [ -d "/usr/local/cuda-12.8" ] || [ -d "/usr/local/cuda-13.0" ]; then
    print_status "✅ CUDA installation found"
    
    # Install CUDA development tools if not present
    if ! dpkg -l | grep -q cuda-toolkit; then
        print_warning "CUDA toolkit not installed via package manager"
        print_warning "Make sure you have the full CUDA toolkit installed"
    fi
else
    print_warning "❌ No CUDA installation found"
    print_status "To install CUDA, visit: https://developer.nvidia.com/cuda-downloads"
fi

# Clone repository if not in it
if [ ! -f "Cargo.toml" ]; then
    print_status "Cloning llama-cpp-rs repository..."
    git clone --recursive https://github.com/utilityai/llama-cpp-rs
    cd llama-cpp-rs
else
    print_status "Already in llama-cpp-rs directory"
    
    # Update submodules
    print_status "Updating submodules..."
    git submodule update --init --recursive
fi

# Test CPU build
print_status "Testing CPU build..."
if cargo build --release --bin simple; then
    print_status "✅ CPU build successful!"
else
    print_error "❌ CPU build failed!"
    exit 1
fi

# Test CUDA build if available
if [ -d "/usr/local/cuda" ] || [ -d "/usr/local/cuda-12.8" ] || [ -d "/usr/local/cuda-13.0" ]; then
    print_status "Testing CUDA build..."
    
    # Set up CUDA environment
    if [ -d "/usr/local/cuda-12.8" ]; then
        export CUDA_ROOT="/usr/local/cuda-12.8"
    elif [ -d "/usr/local/cuda-13.0" ]; then
        export CUDA_ROOT="/usr/local/cuda-13.0"
    else
        export CUDA_ROOT="/usr/local/cuda"
    fi
    
    export PATH="${CUDA_ROOT}/bin:$PATH"
    export CMAKE_ARGS="-DGGML_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_ROOT}"
    
    if cargo build --release --bin simple --features cuda; then
        print_status "✅ CUDA build successful!"
    else
        print_warning "❌ CUDA build failed, but CPU version works"
    fi
fi

print_status "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. For CPU-only inference: ./run_cpu.sh [model_path]"
echo "2. For CUDA inference: ./run_cuda.sh [model_path]"
echo "3. Download a model first if you don't have one"
echo ""
echo "Example model download:"
echo "  mkdir -p models"
echo "  wget -O models/llama-2-7b-chat.Q4_K_M.gguf \\"
echo "    'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf'"