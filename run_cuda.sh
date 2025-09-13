#!/bin/bash

# CUDA Setup and Run Script for llama-cpp-rs
# This script sets up CUDA environment and runs the simple example

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 CUDA Setup and Run Script for llama-cpp-rs${NC}"
echo "=================================================="

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

# Check if CUDA is installed
print_status "Checking CUDA installation..."

CUDA_VERSIONS=()
if [ -d "/usr/local/cuda-12.8" ]; then
    CUDA_VERSIONS+=("12.8")
fi
if [ -d "/usr/local/cuda-13.0" ]; then
    CUDA_VERSIONS+=("13.0")
fi

if [ ${#CUDA_VERSIONS[@]} -eq 0 ]; then
    print_error "No CUDA installation found in /usr/local/"
    print_error "Please install CUDA toolkit first"
    exit 1
fi

# Select CUDA version (prefer 13.0 for better compatibility with newer glibc)
if [[ " ${CUDA_VERSIONS[@]} " =~ " 13.0 " ]]; then
    CUDA_VERSION="13.0"
    CUDA_ROOT="/usr/local/cuda-13.0"
elif [[ " ${CUDA_VERSIONS[@]} " =~ " 12.8 " ]]; then
    CUDA_VERSION="12.8"
    CUDA_ROOT="/usr/local/cuda-12.8"
else
    CUDA_VERSION="${CUDA_VERSIONS[0]}"
    CUDA_ROOT="/usr/local/cuda-${CUDA_VERSION}"
fi

print_status "Using CUDA ${CUDA_VERSION} at ${CUDA_ROOT}"

# Set up CUDA environment
export CUDA_ROOT="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:$LD_LIBRARY_PATH"
export CUDA_PATH="${CUDA_ROOT}"

# Verify CUDA setup
print_status "Verifying CUDA setup..."
if ! command -v nvcc &> /dev/null; then
    print_error "nvcc not found in PATH after setup"
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
print_status "NVCC version: ${NVCC_VERSION}"

# Check GPU
print_status "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read line; do
        print_status "GPU: $line MB"
    done
else
    print_warning "nvidia-smi not found, cannot verify GPU"
fi

# Set CMake variables for CUDA with compatibility fixes
export CMAKE_ARGS="-DGGML_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_ROOT}"
export CUDACXX="${CUDA_ROOT}/bin/nvcc"

# Fix CUDA compatibility issues with newer glibc and system headers
export CUDA_NVCC_FLAGS="--compiler-options -fPIC -Wno-deprecated-gpu-targets"
export CMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets --compiler-options -fPIC"

# Workaround for CUDA/glibc math function conflicts
print_status "Applying CUDA compatibility fixes..."
if [ -f "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" ]; then
    print_warning "Detected potential CUDA/glibc compatibility issue"
    print_status "Using compatibility flags to resolve math function conflicts"
    export CXXFLAGS="${CXXFLAGS} -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__"
fi

# Clean previous build if requested
if [ "$1" = "--clean" ]; then
    print_status "Cleaning previous build..."
    cargo clean
    rm -rf target/release/build/llama-cpp-sys-2-*/
fi

# Default model path (can be overridden)
MODEL_PATH="${2:-/home/cybprgai/.cache/huggingface/hub/models--unsloth--gpt-oss-20b-GGUF/snapshots/c3303d94926e0e2262aacdd0fac4b18e1a29468e/gpt-oss-20b-Q4_K_M.gguf}"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    print_error "Model file not found: $MODEL_PATH"
    print_error "Please provide the correct model path as second argument:"
    print_error "  $0 [--clean] /path/to/your/model.gguf"
    exit 1
fi

print_status "Using model: $MODEL_PATH"

# Build with CUDA
print_status "Building with CUDA support..."
echo "Environment variables:"
echo "  CUDA_ROOT=$CUDA_ROOT"
echo "  PATH=$PATH"
echo "  CMAKE_ARGS=$CMAKE_ARGS"
echo "  CUDACXX=$CUDACXX"
echo "  CMAKE_CUDA_FLAGS=$CMAKE_CUDA_FLAGS"
echo ""

# Try to build with CUDA - attempt multiple strategies
print_status "Compiling with CUDA features (attempt 1/3)..."
if cargo build --release --bin simple --features cuda; then
    print_status "✅ Build successful!"
else
    print_warning "❌ First attempt failed, trying compatibility mode..."
    
    # Try with older C++ ABI and additional compatibility flags
    export CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 -DCMAKE_CUDA_ARCHITECTURES=native"
    print_status "Compiling with CUDA features (attempt 2/3 - compatibility mode)..."
    
    if cargo clean && cargo build --release --bin simple --features cuda; then
        print_status "✅ Build successful with compatibility mode!"
    else
        print_warning "❌ Second attempt failed, trying minimal CUDA build..."
        
        # Try with minimal CUDA configuration
        export CMAKE_ARGS="-DGGML_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_ROOT} -DCMAKE_CUDA_ARCHITECTURES=75"
        print_status "Compiling with CUDA features (attempt 3/3 - minimal config)..."
        
        if cargo clean && cargo build --release --bin simple --features cuda; then
            print_status "✅ Build successful with minimal configuration!"
        else
            print_error "❌ All CUDA build attempts failed!"
            print_error "Trying to diagnose the issue..."
            
            # Check CMake
            if ! command -v cmake &> /dev/null; then
                print_error "CMake not found. Install with: sudo apt install cmake"
            fi
            
            # Check build tools
            if ! command -v gcc &> /dev/null; then
                print_error "GCC not found. Install with: sudo apt install build-essential"
            fi
            
            # Check CUDA compiler
            if ! "${CUDA_ROOT}/bin/nvcc" --version &> /dev/null; then
                print_error "NVCC not working properly"
            fi
            
            print_error "The issue appears to be CUDA ${CUDA_VERSION} compatibility with your system's glibc."
            print_error "Consider using CPU-only version: ./run_cpu.sh"
            print_error "Or try installing CUDA 11.8 which has better compatibility."
            exit 1
        fi
    fi
fi

# Run the example
print_status "Running CUDA-enabled inference..."
echo "Command: cargo run --release --bin simple --features cuda -- --verbose --prompt \"The way to kill a linux process is\" local \"$MODEL_PATH\""
echo ""

time cargo run --release --bin simple --features cuda -- \
    --verbose \
    --prompt "The way to kill a linux process is" \
    local "$MODEL_PATH"

print_status "✅ CUDA run completed!"