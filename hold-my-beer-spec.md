# `gonk` — Zero-dependency GPU inference runtime

## Problem

Running LLM inference on a GPU inside a container requires installing and
correctly configuring a vendor-specific userspace driver stack (Vulkan loader,
ICD JSON files, Mesa or NVIDIA libraries, libglvnd, container toolkits, CDI
specs, environment variables). This stack is fragile, breaks across environments,
and differs by GPU vendor. The kernel drivers and device nodes are always present
— everything above them is accidental complexity.

## Goal

A single static binary that loads a GGUF model and runs inference on any
GPU, with **no dependencies beyond what the GPU driver already provides**.
The only requirements are:

- A Linux kernel with the appropriate GPU kernel module loaded (already the case
  on any machine with a working GPU)
- Access to the GPU device nodes:
  - AMD/Intel: `/dev/dri/renderD*`
  - NVIDIA: `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`
- For NVIDIA: `libcuda.so`, which ships as part of the standard NVIDIA driver
  package (not part of the CUDA Toolkit — installing the `nvidia` driver on any
  distro provides it)

In a container, that means:

```bash
# AMD / Intel
docker run --device /dev/dri my-image smelt ...

# NVIDIA
docker run --device /dev/nvidia0 --device /dev/nvidiactl \
           --device /dev/nvidia-uvm \
           -v /usr/lib/libcuda.so.1:/usr/lib/libcuda.so.1:ro \
           my-image smelt ...
```

No `--runtime=nvidia`. No `--gpus all`. No toolkit. No environment variables.
No Vulkan loader, ICD files, Mesa, libglvnd, or container toolkit.

## MVP scope

Get `smelt` (https://github.com/wegel/smelt) running GPU-accelerated inference
inside a container with nothing but the device nodes mounted. smelt uses
llama.cpp (via the `llama-cpp-2` Rust crate) to run a Qwen3-1.7B Q4_K_M model.

### MVP hardware targets

| Vendor | Architectures | GPU families |
|--------|---------------|--------------|
| AMD    | RDNA2, RDNA3  | RX 6000, RX 7000 |
| NVIDIA | Turing, Ampere, Ada Lovelace | RTX 20xx, 30xx, 40xx |

Intel and older architectures are deferred to post-MVP.

## Architecture

```
┌──────────────────────────────────────────────────┐
│  smelt CLI                                       │
│  (stdin → tokenize → inference → summary)        │
└──────────────┬───────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────┐
│  ggml inference engine                           │
│  (GGUF loader, attention, KV cache, sampling)    │
│  from llama.cpp — used as-is                     │
└──────────────┬───────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────┐
│  gonk backend (replaces ggml-vulkan)            │
│  tensor op dispatch → selects pre-compiled       │
│  native GPU kernel for detected architecture     │
└──────────────┬───────────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌──────▼──────────────────┐
│ AMD runtime │  │ NVIDIA runtime          │
│             │  │                         │
│ amdgpu DRM  │  │ dlopen libcuda.so       │
│ ioctls via  │  │ (ships with the driver) │
│ /dev/dri    │  │ via /dev/nvidia*        │
└─────────────┘  └─────────────────────────┘
```

## Key design decisions

### 1. Pre-compiled native GPU kernels (no runtime shader compilation)

llama.cpp's Vulkan backend has ~200 SPIR-V shader variants covering all
tensor operations needed for transformer inference (~25 core ops × data
types × tile sizes). These are already written and battle-tested.

At **build time**, we compile each SPIR-V shader to native ISA for every
target architecture:

| Target | Compiler | Input | Output |
|--------|----------|-------|--------|
| AMD RDNA2 | ACO (from Mesa, build-time only) | SPIR-V | RDNA2 binary (ELF) |
| AMD RDNA3 | ACO | SPIR-V | RDNA3 binary (ELF) |
| NVIDIA Turing | ptxas (NVIDIA offline compiler) | SPIR-V → PTX (via naga or spirv-cross) | SASS |
| NVIDIA Ampere | ptxas | SPIR-V → PTX | SASS |
| NVIDIA Ada | ptxas | SPIR-V → PTX | SASS |

The compiled blobs are embedded in the final binary as static byte arrays.
~200 variants × ~5 architectures × ~4 KB average = ~4 MB total. Negligible.

At **runtime**, zero compilation. The binary detects the GPU architecture
via sysfs PCI device ID and selects the correct blob set.

This means Mesa compilers (ACO, NAK, intel_compiler) are **build-time
dependencies only**. They do not ship in the final binary.

### 2. Direct kernel interface (no Vulkan, no loader, no ICD)

The runtime talks directly to the kernel driver via ioctls:

**AMD (amdgpu DRM interface):**
- `DRM_IOCTL_AMDGPU_GEM_CREATE` — allocate GPU buffer objects
- `DRM_IOCTL_AMDGPU_GEM_MMAP` — map buffers for CPU access
- `DRM_IOCTL_AMDGPU_BO_VA` — set up GPU virtual address mappings
- `DRM_IOCTL_AMDGPU_CS` — submit command buffer (PM4 packets containing
  the pre-compiled compute shader dispatch)
- `DRM_IOCTL_AMDGPU_WAIT_CS` — wait for completion

These are stable kernel UAPI. Mesa's radv uses the same ioctls. The PM4
command stream format for compute dispatch is well-documented in AMD's
open register specs.

**NVIDIA (libcuda.so — the driver's command submission interface):**

NVIDIA's kernel module (`nvidia.ko`) does not expose a stable, documented
ioctl interface for command submission. Unlike AMD's amdgpu, you cannot
build command buffers yourself and submit them via ioctls — the format is
undocumented and changes between driver versions.

However, `libcuda.so` provides a stable, backwards-compatible command
submission API that has been unchanged for over 15 years. It is **not**
part of the CUDA Toolkit — it ships with the standard NVIDIA driver
package. On Arch, `pacman -S nvidia` installs it. Anywhere `/dev/nvidia0`
exists, `libcuda.so` is present, because they come from the same package.

We use it not as "CUDA programming" but as NVIDIA's stable interface for
talking to their kernel module. The entire API surface we need:

```
cuInit                — initialize
cuDeviceGet           — select GPU
cuCtxCreate           — create context
cuMemAlloc / cuMemFree — GPU buffer management
cuMemcpyHtoD / DtoH  — data transfer
cuModuleLoadData      — load pre-compiled SASS binary (cubin)
cuModuleGetFunction   — get kernel entry point from loaded module
cuLaunchKernel        — submit kernel with grid/block dims + arguments
cuCtxSynchronize      — wait for completion
```

That's ~10 functions, resolved via dlopen at runtime. No CUDA programming
model, no nvcc, no CUDA Toolkit, no CUDA headers. Just a command
submission interface to the GPU, backed by pre-compiled native kernels.

For containers, `libcuda.so` can be bind-mounted from the host alongside
the device nodes. This is a single file, at a known path, with a stable
interface — fundamentally simpler than the current nvidia-container-toolkit
stack with its ICD discovery, CDI specs, and environment variables.

### 3. GPU detection via sysfs

No Vulkan instance creation, no device enumeration API. Just:

```
/sys/class/drm/renderD128/device/vendor  → 0x1002 (AMD) / 0x8086 (Intel) / 0x10de (NVIDIA)
/sys/class/drm/renderD128/device/device  → PCI device ID → architecture lookup table
```

For NVIDIA proprietary (no DRM render node for compute):
```
/sys/bus/pci/devices/*/vendor → 0x10de
/sys/bus/pci/devices/*/device → PCI device ID
```

A static lookup table maps PCI device IDs to architecture names, which
select the correct pre-compiled shader blob set.

### 4. Integration with ggml

The gonk backend implements ggml's backend interface (the same interface
that `ggml-vulkan`, `ggml-cuda`, and `ggml-metal` implement). This means:

- ggml's graph scheduler dispatches tensor operations to gonk
- gonk maps each op to a pre-compiled kernel + buffer bindings
- gonk submits the kernel via the appropriate vendor runtime
- ggml handles everything else (model loading, tokenization, sampling)

smelt continues to use `llama-cpp-2` (Rust bindings to llama.cpp). The
only change is that llama.cpp is built with the gonk backend instead of
ggml-vulkan/ggml-cuda.

## Build pipeline

```
                    llama.cpp GLSL shaders
                           │
                    glslangValidator
                           │
                       SPIR-V blobs
                           │
              ┌────────────┼────────────┐
              │            │            │
             ACO       spirv-cross    spirv-cross
         (Mesa, build    + ptxas      + ptxas
          time only)   (sm_75)       (sm_86, sm_89)
              │            │            │
          RDNA2/3       Turing       Ampere/Ada
          ELF blobs    SASS cubins   SASS cubins
              │            │            │
              └────────────┼────────────┘
                           │
                    embed as static
                    arrays in binary
                           │
              ┌────────────┼────────────┐
              │            │            │
          amd_rt.rs    nvidia_rt.rs   detect.rs
         (DRM ioctls)  (dlopen        (sysfs probe)
              │         libcuda.so)         │
              │            │            │
              └────────────┼────────────┘
                           │
                      gonk library
                      (ggml backend)
                           │
                      llama-cpp-2
                           │
                        smelt
```

### Build dependencies (not shipped at runtime)

- Mesa source (for ACO compiler) — or a standalone ACO extraction
- ptxas (NVIDIA's offline shader compiler — part of the CUDA Toolkit, but
  only needed at build time to produce cubins; not needed at runtime)
- glslangValidator or shaderc (for SPIR-V compilation)
- spirv-cross (for SPIR-V → PTX translation)
- Rust toolchain
- C/C++ compiler (for llama.cpp)

### Runtime dependencies

- Linux kernel with GPU driver module loaded (`amdgpu` / `nvidia`)
- GPU device nodes accessible
- For NVIDIA: `libcuda.so` (ships with the NVIDIA driver package, not the
  CUDA Toolkit — present on any system with a working NVIDIA GPU)

That's it.

## Components and work estimates

### 1. Shader pre-compilation pipeline
Build system that takes llama.cpp's GLSL shaders, compiles through SPIR-V
to native ISA for each target architecture, and embeds the results.

**Work: 3-4 weeks**

Key tasks:
- Extract and catalog all shader variants from llama.cpp's vulkan-shaders/
- Script the SPIR-V → PTX → SASS path for NVIDIA targets
- Script the SPIR-V → RDNA binary path via ACO
- Embed compiled blobs with architecture tags
- Verify binary correctness against reference Vulkan output

### 2. GPU detection module
Scan sysfs, identify GPU vendor and architecture, select correct blob set.

**Work: 1 week**

Key tasks:
- Enumerate /sys/class/drm/renderD* and /sys/bus/pci/devices/*
- PCI device ID → architecture lookup table (source: Mesa's pci_id_driver_map,
  NVIDIA's CUDA device table)
- Handle multi-GPU (pick first capable device, or allow selection)

### 3. AMD amdgpu submission runtime
Open render node, allocate buffers, build PM4 command streams, submit
compute dispatches, synchronize.

**Work: 5-6 weeks**

Key tasks:
- Open /dev/dri/renderD* for the detected AMD GPU
- Buffer management: GEM create, mmap, VA mapping
- Build PM4 command packets for compute dispatch (COMPUTE_PFP packets:
  set shader, set buffers, set dispatch dimensions, execute)
- Command buffer submission via DRM_IOCTL_AMDGPU_CS
- Fence wait for synchronization
- Implement ggml backend buffer and compute interfaces

Reference: Mesa's `src/amd/vulkan/radv_cmd_buffer.c` and
`src/amd/common/ac_pm4.c` for PM4 packet construction.

### 4. NVIDIA submission runtime
Load the driver's command submission library and use it to manage buffers
and launch pre-compiled kernels.

**Work: 3-4 weeks**

Key tasks:
- dlopen libcuda.so from known paths (/usr/lib/, /usr/lib64/,
  /usr/lib/x86_64-linux-gnu/, etc.)
- Resolve ~10 symbols: cuInit, cuDeviceGet, cuCtxCreate, cuMemAlloc,
  cuMemcpyHtoD, cuMemcpyDtoH, cuModuleLoadData, cuModuleGetFunction,
  cuLaunchKernel, cuCtxSynchronize
- Load pre-compiled SASS cubins via cuModuleLoadData
- Map ggml tensor ops to kernel launches with correct argument bindings
- Implement ggml backend buffer and compute interfaces
- Handle library-not-found gracefully (fall back to CPU with clear message)

### 5. ggml backend integration
Wire gonk into llama.cpp's backend registration system so ggml's graph
scheduler dispatches ops to it.

**Work: 5-6 weeks**

Key tasks:
- Implement `ggml_backend_gonk_init()`
- Implement buffer type (alloc, free, get_base, set, get, copy)
- Implement compute graph execution (iterate ops, map to kernels, dispatch)
- Handle ops that need CPU fallback (anything without a GPU kernel)
- Ensure numerical correctness against ggml-vulkan reference output
- Adapt llama-cpp-2 Rust crate to expose the new backend as a feature flag

### 6. smelt integration
Modify smelt's Cargo.toml to support an `gonk` feature that uses the new
backend instead of vulkan/cuda.

**Work: 1-2 weeks**

Key tasks:
- Add `gonk` feature flag to smelt
- Build and test container image with only device node mounts
- Write Containerfile for AMD and NVIDIA variants
- Verify end-to-end: pipe → tokenize → GPU inference → summary output

### 7. Testing and validation

**Work: 4-6 weeks (overlapping with above)**

Key tasks:
- Numerical correctness: compare output logits against ggml-vulkan for the
  same model + prompt across all target GPUs
- Performance: measure tokens/sec vs ggml-vulkan baseline (target: within 20%)
- Container testing: verify on Docker and Podman, rootless and rootful
- Edge cases: model too large for VRAM (graceful CPU fallback),
  multi-GPU detection, driver version mismatches

## Total estimate

**5-7 months for one experienced systems developer** familiar with GPU
driver internals and ggml/llama.cpp.

**8-12 months** for a developer learning the GPU driver interfaces as they go.

The highest-risk component is the AMD PM4 command submission (component 3)
— it requires understanding AMD's hardware command processor protocol. The
information exists in Mesa's source and AMD's open documentation, but it's
dense. NVIDIA's path (component 4) is lower risk because we're using
`libcuda.so`'s stable, documented API rather than raw ioctls — it's
essentially 10 function calls with a 15+ year track record of backwards
compatibility.

## What this is NOT

- Not a general-purpose GPU compute framework. It runs inference kernels
  for transformer models. That's it.
- Not a Vulkan replacement. It doesn't implement any Vulkan API surface.
- Not "using CUDA." We don't write CUDA code, don't use nvcc at runtime,
  don't link the CUDA Toolkit, and don't use the CUDA programming model.
  We use `libcuda.so` purely as NVIDIA's command submission interface —
  the same way we use amdgpu DRM ioctls for AMD. It happens to be called
  "CUDA Driver API" but it's just the driver's way of accepting work.
- Not portable to non-Linux. The DRM ioctl interface is Linux-specific.
  macOS would need a Metal backend. Windows would need a different approach
  entirely.
- Not a new shader language or compiler. It reuses llama.cpp's existing
  shaders and existing compilers (ACO, ptxas) at build time only.

## Open questions

1. **ACO extraction**: Can ACO be built standalone outside Mesa, or do we
   need to vendor a Mesa snapshot and build a subset? Initial investigation
   suggests ACO has a relatively clean boundary via `ac_nir_to_asm()` but
   pulling it out of Mesa's build system needs prototyping.

2. **SPIR-V → PTX translation quality**: spirv-cross produces PTX from
   SPIR-V, but llama.cpp's shaders use Vulkan-specific features
   (push constants, descriptor sets, specialization constants). These need
   to be mapped to CUDA kernel argument equivalents. May require a custom
   translation pass rather than a straight spirv-cross invocation.

3. **cubin portability**: SASS cubins produced by ptxas are
   architecture-specific (sm_75, sm_86, sm_89). We need to confirm that
   cubins loaded via cuModuleLoadData work with only libcuda.so present —
   no other CUDA Toolkit components at runtime. Initial evidence suggests
   yes (libcuda.so handles cubin loading natively), but needs verification.

4. **libcuda.so version compatibility**: While the CUDA Driver API is
   backwards compatible, we should verify that cubins compiled with a
   newer ptxas work on an older libcuda.so (or vice versa). The general
   contract is that the driver is forwards-compatible with older cubins,
   but not the reverse — so we should compile cubins against the oldest
   supported SM version for each architecture.

5. **amdgpu kernel UAPI stability**: While DRM ioctls are nominally stable
   UAPI, the PM4 packet format and register definitions could shift between
   GPU generations. Need to confirm that RDNA2 and RDNA3 use compatible
   dispatch mechanisms (they should — radv handles both).

6. **ggml backend API stability**: ggml's backend interface is evolving.
   Building against a specific llama.cpp version and tracking upstream
   changes is an ongoing maintenance cost.

## Success criteria

smelt, built with `--features gonk`, runs Qwen3-1.7B Q4_K_M inference
on an NVIDIA RTX 3090 and an AMD RX 7900 inside Docker containers with
only device nodes and (for NVIDIA) a single bind-mounted `libcuda.so`.

```bash
# AMD — zero host dependencies beyond the kernel driver
docker run --device /dev/dri my-image smelt ...

# NVIDIA — device nodes + the driver's command submission library
docker run --device /dev/nvidia0 --device /dev/nvidiactl \
           --device /dev/nvidia-uvm \
           -v /usr/lib/libcuda.so.1:/usr/lib/libcuda.so.1:ro \
           my-image smelt ...
```

No nvidia-container-toolkit. No CDI. No `--runtime=nvidia`. No `--gpus all`.
No Vulkan loader, ICD JSON files, Mesa, or libglvnd installed in the
container. No environment variables.

Output matches the Vulkan backend numerically. Performance is within 20%
of the Vulkan backend.

