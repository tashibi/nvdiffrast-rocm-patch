# nvdiffrast-rocm-patch
Fixes and patches for NVlabs/nvdiffrast to support AMD ROCm 7.1 and Wave64 architectures (gfx1100, gfx1201).
# nvdiffrast Patch for AMD ROCm 7.1+ (Wave64 Support)

This repository provides a comprehensive patch to make [NVlabs/nvdiffrast](https://github.com) fully compatible with **AMD GPUs** using **ROCm 7.1** and newer architectures like **RDNA 3/4** (gfx1100, gfx1201).

## 🧩 Why is this needed?
The original `nvdiffrast` is built for NVIDIA CUDA. Standard "hipify" conversion fails on newer AMD cards because:
1. **Wavefront Size:** Newer AMD GPUs use **Wave64**, requiring 64-bit lane masks, while CUDA code uses 32-bit.
2. **NVIDIA ASM:** The code contains PTX assembly that doesn't exist in ROCm.
3. **C++ Strictness:** Modern ROCm compilers (Clang 20+) are stricter about narrowing conversions and namespaces.

## 🚀 What this patch fixes
- ✅ **Wave64 Support:** Converts all 32-bit masks (`0xffffffffu`) to 64-bit (`0xffffffffffffffffull`).
- ✅ **ASM Porting:** Replaces `vmin`, `vmax`, `slct`, `prmt`, and `bfind` with cross-platform HIP intrinsics.
- ✅ **PyTorch Compatibility:** Fixes `OptionalCUDAGuard` namespaces and narrowing errors in `torch_antialias.cpp`.
- ✅ **Header Aliasing:** Correctly links ROCm headers to expected CUDA paths.

## 🛠️ Quick Start

### 1. Install System Dependencies
```bash
sudo apt update && sudo apt install -y hipsparse-dev hipblas-dev rocthrust-dev hipcub-dev
```

### 2. Clone and Patch
```bash
# Clone the original nvdiffrast
git clone https://github.com
cd nvdiffrast

# Download and run this patch
wget https://githubusercontent.com
chmod +x patch_rocm.sh
./patch_rocm.sh
```

### 3. Build and Install
```bash
rm -rf build/
export PYTORCH_ROCM_ARCH=gfx1201  # Change to your arch (gfx1100 for 7900XTX, etc.)
export FORCE_CUDA=1
python3 setup.py install
```

## 🧪 Verification
```python
import torch
import nvdiffrast.torch as dr
ctx = dr.RasterizeCudaContext()
print("nvdiffrast successfully loaded on ROCm!")
```

## 💡 InstantMesh Tips
If you use this for **InstantMesh** on a 16GB card:
- Use `grid_res: 96` or `128` in your config.
- Set `export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128` to avoid OOM.
