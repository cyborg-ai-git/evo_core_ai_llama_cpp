# 🦙 [llama-cpp-rs][readme] &emsp; [![Docs]][docs.rs] [![Latest Version]][crates.io] [![Lisence]][crates.io]

[Docs]: https://img.shields.io/docsrs/llama-cpp-2.svg

[Latest Version]: https://img.shields.io/crates/v/llama-cpp-2.svg

[crates.io]: https://crates.io/crates/llama-cpp-2

[docs.rs]: https://docs.rs/llama-cpp-2

[Lisence]: https://img.shields.io/crates/l/llama-cpp-2.svg

[llama-cpp-sys]: https://crates.io/crates/llama-cpp-sys-2

[utilityai]: https://utilityai.ca

[readme]: https://github.com/utilityai/llama-cpp-rs/tree/main/llama-cpp-2

This is the home for [llama-cpp-2][crates.io]. It also contains the [llama-cpp-sys] bindings which are updated semi-regularly
and in sync with [llama-cpp-2][crates.io].

This project was created with the explict goal of staying as up to date as possible with llama.cpp, as a result it is
dead simple, very close to raw bindings, and does not follow semver meaningfully.

Check out the [docs.rs] for crate documentation or the [readme] for high level information about the project.

## Quick Start

### 🚀 Automated Setup (Recommended)

We provide automated scripts for easy setup and execution:

```bash
# Clone the repository
git clone --recursive https://github.com/utilityai/llama-cpp-rs
cd llama-cpp-rs

# Run automated setup (installs dependencies)
./setup.sh

# Run with CPU (stable and fast)
./run_cpu.sh [path/to/your/model.gguf]

# Run with CUDA (if available)
./run_cuda.sh [path/to/your/model.gguf]
```

### 📋 Manual Setup

If you prefer manual setup:

```bash
# Install dependencies
sudo apt update
sudo apt install -y build-essential cmake pkg-config libssl-dev git curl

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Clone and build
git clone --recursive https://github.com/utilityai/llama-cpp-rs
cd llama-cpp-rs
```

### 💻 CPU-Only Usage (Recommended - Stable)

```bash
# Basic usage
cargo run --release --bin simple -- --prompt "The way to kill a linux process is" local /path/to/your/model.gguf

# With verbose output for debugging
cargo run --release --bin simple -- --verbose --prompt "Hello world" local /path/to/your/model.gguf

# With custom parameters
cargo run --release --bin simple -- --prompt "Hello world" --n-len 50 --ctx-size 4096 --threads 8 local /path/to/your/model.gguf

# Using Hugging Face model download
cargo run --release --bin simple -- --prompt "Hello" hf-model TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf
```

### 🚀 CUDA GPU Acceleration (✅ Working!)

**Prerequisites:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x or 13.x installed
- Updated NVIDIA drivers (may need driver update for runtime)

```bash
# Automated CUDA setup and run (recommended)
./run_cuda.sh [path/to/your/model.gguf]

# Manual CUDA setup
export CUDA_ROOT=/usr/local/cuda-13.0  # or cuda-12.8
export PATH=$CUDA_ROOT/bin:$PATH
export CMAKE_ARGS="-DGGML_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOT"

# Build and run with CUDA
cargo build --release --bin simple --features cuda
cargo run --release --bin simple --features cuda -- --verbose --prompt "Hello" local /path/to/your/model.gguf
```

> **📝 Note**: CUDA build now works! If you see "CUDA driver version is insufficient", update your NVIDIA drivers. The build will still work and fall back to CPU if needed.

## Available Options

The simple example supports various command-line options:

- `--prompt <PROMPT>` - The text prompt to generate from
- `--verbose` or `-v` - Enable detailed logging and model information
- `--n-len <N_LEN>` - Set total length of prompt + output in tokens (default: 32)
- `--ctx-size <CTX_SIZE>` - Set context window size (default: from model)
- `--threads <THREADS>` - Number of CPU threads to use
- `--seed <SEED>` - Random seed for reproducible generation

<details>
<summary>CPU Example Output (Normal)</summary>
<pre>
n_len = 32, n_ctx = 2048, k_kv_req = 32

The way to kill a linux process is to send it a signal. The most common signals are SIGTERM (15) and SIGKILL (9). The default

decoded 25 tokens in 3.91 s, speed 6.40 t/s

load time = 313053.91 ms
prompt eval time = 0.00 ms / 8 tokens (0.00 ms per token, inf tokens per second)
eval time = 0.00 ms / 24 runs (0.00 ms per token, inf tokens per second)
</pre>
</details>

<details>
<summary>CPU Example Output (Verbose)</summary>
<pre>
2025-09-12T10:05:07.436518Z  INFO load_from_file: llama-cpp-2: loaded meta data with 37 key-value pairs and 459 tensors from model.gguf (version GGUF V3)
2025-09-12T10:05:07.590599Z  INFO load_from_file: llama-cpp-2: file format = GGUF V3 (latest)
2025-09-12T10:05:07.590606Z  INFO load_from_file: llama-cpp-2: file type   = Q4_K - Medium
2025-09-12T10:05:07.590621Z  INFO load_from_file: llama-cpp-2: file size   = 10.81 GiB (4.44 BPW)
2025-09-12T10:05:08.188351Z  INFO load_from_file: llama-cpp-2: arch             = gpt-oss
2025-09-12T10:05:08.188371Z  INFO load_from_file: llama-cpp-2: n_ctx_train      = 131072
2025-09-12T10:05:08.188377Z  INFO load_from_file: llama-cpp-2: n_embd           = 2880
2025-09-12T10:05:08.188382Z  INFO load_from_file: llama-cpp-2: n_layer          = 24
2025-09-12T10:05:08.188544Z  INFO load_from_file: llama-cpp-2: model params     = 20.91 B
2025-09-12T10:05:08.188487Z  INFO load_from_file: llama-cpp-2: n_expert         = 32
2025-09-12T10:05:08.188492Z  INFO load_from_file: llama-cpp-2: n_expert_used    = 4

n_len = 32, n_ctx = 2048, k_kv_req = 32

The way to kill a linux process is to send it a signal. The most common signals are SIGTERM (15) and SIGKILL (9). The default

decoded 25 tokens in 4.79 s, speed 5.21 t/s

load time = 2821.60 ms
prompt eval time = 0.00 ms / 8 tokens (0.00 ms per token, inf tokens per second)
eval time = 0.00 ms / 24 runs (0.00 ms per token, inf tokens per second)
</pre>
</details>

<details>
<summary>CUDA Example Output (When Working)</summary>
<pre>
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
llm_load_tensors:      CUDA0 buffer size =  3820.94 MiB
llm_load_tensors:        CPU buffer size =    70.31 MiB

The way to kill a linux process is to send it a SIGKILL signal.
The way to kill a windows process is to send it a S

decoded 24 tokens in 0.23 s, speed 105.65 t/s

load time = 727.50 ms
sample time = 0.46 ms / 24 runs (0.02 ms per token, 51835.85 tokens per second)
prompt eval time = 68.52 ms / 9 tokens (7.61 ms per token, 131.35 tokens per second)
eval time = 225.70 ms / 24 runs (9.40 ms per token, 106.34 tokens per second)
total time = 954.18 ms
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   4:                          llama.block_count u32              = 32
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 11008
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 32
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 15
llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  17:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  18:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q4_K:  193 tensors
llama_model_loader: - type q6_K:   33 tensors
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V2
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 4096
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 32
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 4096
llm_load_print_meta: n_embd_v_gqa     = 4096
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 11008
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 6.74 B
llm_load_print_meta: model size       = 3.80 GiB (4.84 BPW) 
llm_load_print_meta: general.name     = LLaMA v2
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.22 MiB
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
llm_load_tensors:      CUDA0 buffer size =  3820.94 MiB
llm_load_tensors:        CPU buffer size =    70.31 MiB
..................................................................................................
Loaded "/home/marcus/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-GGUF/snapshots/b4e04e128f421c93a5f1e34ac4d7ca9b0af47b80/llama-2-7b.Q4_K_M.gguf"
llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  1024.00 MiB
llama_new_context_with_model: KV self size  = 1024.00 MiB, K (f16):  512.00 MiB, V (f16):  512.00 MiB
llama_new_context_with_model:  CUDA_Host input buffer size   =    13.02 MiB
ggml_gallocr_reserve_n: reallocating CUDA0 buffer from size 0.00 MiB to 164.01 MiB
ggml_gallocr_reserve_n: reallocating CUDA_Host buffer from size 0.00 MiB to 8.00 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   164.01 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =     8.00 MiB
llama_new_context_with_model: graph splits (measure): 3
n_len = 32, n_ctx = 2048, k_kv_req = 32

The way to kill a linux process is to send it a SIGKILL signal.
The way to kill a windows process is to send it a S

decoded 24 tokens in 0.23 s, speed 105.65 t/s

load time = 727.50 ms
sample time = 0.46 ms / 24 runs (0.02 ms per token, 51835.85 tokens per second)
prompt eval time = 68.52 ms / 9 tokens (7.61 ms per token, 131.35 tokens per second)
eval time = 225.70 ms / 24 runs (9.40 ms per token, 106.34 tokens per second)
total time = 954.18 ms
</pre>
</details>

## 📁 Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup.sh` | Install all dependencies and test builds | `./setup.sh` |
| `run_cpu.sh` | Run CPU-only inference | `./run_cpu.sh [model_path]` |
| `run_cuda.sh` | Run CUDA-accelerated inference | `./run_cuda.sh [model_path]` |

### Script Features:
- ✅ **Automatic dependency detection**
- ✅ **CUDA version detection and setup**
- ✅ **Colored output and progress indicators**
- ✅ **Error handling and diagnostics**
- ✅ **Performance monitoring**

## 🛠️ Troubleshooting

### CUDA Issues
```bash
# Check CUDA installation
ls /usr/local/cuda*
nvcc --version

# Check GPU and drivers
nvidia-smi

# Clean build and retry
cargo clean
./run_cuda.sh --clean [model_path]

# Update NVIDIA drivers (if needed)
sudo apt update
sudo apt install nvidia-driver-535  # or latest version
sudo reboot
```

**Common CUDA Messages:**
- `"CUDA driver version is insufficient"` → Update NVIDIA drivers
- `"failed to initialize CUDA"` → Check `nvidia-smi` works
- `"CMake CUDA compilation failed"` → Try `./run_cuda.sh --clean`

### Common Issues
| Issue | Solution |
|-------|----------|
| **Build fails** | Run `./setup.sh` to install dependencies |
| **CUDA not found** | Set `CUDA_ROOT` environment variable |
| **Out of memory** | Reduce `--ctx-size` or use smaller model |
| **Slow performance** | Adjust `--threads` to match CPU cores |
| **Model not found** | Check file path and permissions |

### Performance Tips
- 🎯 **Use Q4_K_M quantization** for best speed/quality balance
- 🧠 **Set `--ctx-size`** based on available RAM (2048-8192 typical)
- ⚡ **Use `--threads`** equal to your CPU core count
- 📊 **Monitor with `--verbose`** to see detailed performance metrics
- 🚀 **CUDA acceleration** can be 10-50x faster than CPU

## 📥 Getting Models

### Download Example Models
```bash
# Create models directory
mkdir -p models

# Download Llama 2 7B (recommended for testing)
wget -O models/llama-2-7b-chat.Q4_K_M.gguf \
  'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf'

# Download smaller model for quick testing
wget -O models/tinyllama-1.1b.Q4_K_M.gguf \
  'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'

# Run with downloaded model
./run_cpu.sh models/llama-2-7b-chat.Q4_K_M.gguf
```

## 🔧 Development

### Building from Source
```bash
# Ensure submodules are updated
git clone --recursive https://github.com/utilityai/llama-cpp-rs
cd llama-cpp-rs

# Or update existing clone
git submodule update --init --recursive

# Build CPU version
cargo build --release --bin simple

# Build CUDA version (requires CUDA toolkit)
export CUDA_ROOT=/usr/local/cuda-12.8
cargo build --release --bin simple --features cuda
```

### Environment Variables
| Variable | Purpose | Example |
|----------|---------|---------|
| `CUDA_ROOT` | CUDA installation path | `/usr/local/cuda-12.8` |
| `CMAKE_ARGS` | CMake build arguments | `-DGGML_CUDA=ON` |
| `LLAMA_BUILD_SHARED_LIBS` | Use dynamic linking | `1` |

### Build Features
- `cuda` - Enable CUDA GPU acceleration
- `openmp` - Enable OpenMP parallelization
- `dynamic-link` - Use shared libraries instead of static linking

## 🚀 Performance Comparison

| Configuration | Speed (tokens/sec) | Memory Usage | Setup Difficulty |
|---------------|-------------------|--------------|------------------|
| **CPU (Static)** | 5-15 t/s | ~11GB RAM | ⭐ Easy |
| **CPU (Optimized)** | 10-25 t/s | ~11GB RAM | ⭐⭐ Medium |
| **CUDA GPU** | 50-200+ t/s | ~8GB VRAM | ⭐⭐⭐ Hard |

*Performance varies by model size, hardware, and quantization level*