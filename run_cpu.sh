#!/bin/bash

# CPU-only Run Script for llama-cpp-rs
# This script runs the simple example with CPU-only inference

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 CPU-only Run Script for llama-cpp-rs${NC}"
echo "============================================="

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

# Clean previous build if requested
if [ "$1" = "--clean" ]; then
    print_status "Cleaning previous build..."
    cargo clean
    shift
fi

# Default model path (can be overridden)
MODEL_PATH="${1:-/home/cybprgai/.cache/huggingface/hub/models--unsloth--gpt-oss-20b-GGUF/snapshots/c3303d94926e0e2262aacdd0fac4b18e1a29468e/gpt-oss-20b-Q4_K_M.gguf}"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    print_error "Model file not found: $MODEL_PATH"
    print_error "Please provide the correct model path as argument:"
    print_error "  $0 [--clean] /path/to/your/model.gguf"
    exit 1
fi

print_status "Using model: $MODEL_PATH"

# Check system info
print_status "System information:"
echo "  CPU cores: $(nproc)"
echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "  Architecture: $(uname -m)"
echo ""

# Build CPU version
print_status "Building CPU-only version..."
if cargo build --release --bin simple; then
    print_status "✅ Build successful!"
else
    print_error "❌ Build failed!"
    exit 1
fi

# Run the example
print_status "Running CPU inference..."
echo "Command: cargo run --release --bin simple -- --verbose --prompt \"The way to kill a linux process is\" local \"$MODEL_PATH\""
echo ""

time cargo run --release --bin simple -- \
    --verbose \
    --prompt "The way to kill a linux process is" \
    local "$MODEL_PATH"

print_status "✅ CPU run completed!"