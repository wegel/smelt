# smelt

Compress verbose command output into concise summaries using a local LLM. No API keys, no external services — runs entirely on your machine.

```
cargo build --release 2>&1 | smelt
```

> smelt v0.1.0 compiled, optimized, release profile in 0.92s.

## How it works

smelt reads stdin, feeds it through a small local model (Qwen2.5 1.5B, Q4 quantized), and outputs a one-sentence summary. The model is automatically downloaded on first run (~1.1GB) to `~/.local/share/smelt/models/`.

Inference runs on GPU via Vulkan, CUDA, or Metal. On CPU it works but is significantly slower.

## Usage

Pipe any command through smelt:

```sh
npm install 2>&1 | smelt
cargo test 2>&1 | smelt
kubectl get pods -A | smelt
git log --oneline -50 | smelt
```

### Strategies

When input exceeds the model's context window, smelt has two strategies for handling it:

**`--last`** (default)
Keeps the last N lines that fit in context, summarizes once. Fast. Best when the end of output has the verdict (build result, test pass/fail, error messages).

**`--rolling`**
Processes the input in chunks sequentially, carrying a running summary forward into each next chunk. Slower (one inference per chunk) but captures information from the entire output.

```sh
# Long build log — just care about the result
make 2>&1 | smelt --last

# Long test suite — want to know about all failures
pytest -v 2>&1 | smelt --rolling
```

### Passthrough

Use `--head` and `--tail` to pass through raw lines without summarization, like the unix utilities:

```sh
# First and last 5 lines
some-command 2>&1 | smelt --head 5 --tail 5

# Just the last 10 lines
some-command 2>&1 | smelt --tail 10
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--last` | yes | Summarize only the tail of the input |
| `--rolling` | | Rolling summary over the full input |
| `--head N` | | Passthrough first N lines (no LLM) |
| `--tail N` | | Passthrough last N lines (no LLM) |
| `--prompt TEXT` | "Summarize this command output:" | Custom instruction prompt |
| `--ctx-size N` | 8192 | Context window size in tokens |
| `-v, --verbose` | | Show timing and progress on stderr |

## Performance

Benchmarked on RTX 2070 + GTX 1060 with Vulkan:

| Input | Strategy | Time |
|-------|----------|------|
| 10 lines | last | 1.2s |
| 150 lines | last | 2.7s |
| 150 lines | rolling | 2.9s |

Model load is ~0.8s (cached). Inference runs at ~50 tokens/s on GPU.

## Model

Uses [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) (Q4_K_M quantization, ~1.1GB). Downloaded automatically to `~/.local/share/smelt/models/` on first run.

## Install

```sh
# Vulkan (Linux/Windows with NVIDIA, AMD, or Intel GPUs)
cargo install --path . --features vulkan

# CUDA (NVIDIA only)
cargo install --path . --features cuda

# Metal (macOS)
cargo install --path . --features metal

# CPU only (no GPU acceleration)
cargo install --path .
```

### System dependencies

**Arch Linux (Vulkan):**
```sh
sudo pacman -S vulkan-headers shaderc vulkan-icd-loader
```

**Ubuntu/Debian (Vulkan):**
```sh
sudo apt install libvulkan-dev glslang-tools
```

**CUDA:** Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

## License

MIT
