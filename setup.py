"""
fused-ssim build system.

Multi-backend (CUDA, MPS, SYCL, ROCm/HIP) build with environment-variable
controls for GPU architecture, parallel jobs, debug symbols, and compile flags.

CUDA architecture handling
--------------------------
- If TORCH_CUDA_ARCH_LIST is set, it is honored (cross-compile / build-once).
- If TORCH_CUDA_ARCH_LIST is unset AND a local GPU is available, PyTorch
  auto-detects the local GPU's compute capability (single-arch, smallest
  binary, fastest compile).
- If TORCH_CUDA_ARCH_LIST is unset AND no local GPU is available, the build
  raises a clear error. This prevents the common footgun of accidentally
  shipping a bloated multi-arch binary from a CI machine without a GPU.

Build-time environment variables
--------------------------------
BUILD_NO_CUDA=1            Skip extension compilation entirely (no GPU code in wheel).
WITH_SYMBOLS=1             Keep debug symbols in compiled .so (don't strip with -s).
LINE_INFO=1                Add -lineinfo to nvcc for line-level profiling.
MAX_JOBS=N                 Parallel compile jobs (default: cpu_count - 2, min 2).
NVCC_FLAGS="..."           Extra flags appended to nvcc (space-separated).
NO_FAST_MATH=1             Disable nvcc --use_fast_math.
VERBOSE=1                  Verbose build output (passed to BuildExtension).
"""

import os
import sys

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"
NO_FAST_MATH = os.getenv("NO_FAST_MATH", "0") == "1"
VERBOSE = os.getenv("VERBOSE", "0") == "1"
NVCC_FLAGS_EXTRA = os.getenv("NVCC_FLAGS", "")

_need_to_unset_max_jobs = False
if not os.getenv("MAX_JOBS"):
    _default_max_jobs = max((os.cpu_count() or 1) - 2, 2)
    os.environ["MAX_JOBS"] = str(_default_max_jobs)
    _need_to_unset_max_jobs = True
    print(f"[fused-ssim] MAX_JOBS not set, defaulting to {os.environ['MAX_JOBS']}")

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stderr.reconfigure(line_buffering=True)


def log(msg: str) -> None:
    print(msg)
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------
def _fail_if_no_gpu_and_no_arch_list() -> None:
    """If TORCH_CUDA_ARCH_LIST is unset, require a local GPU for auto-detect.

    Without this, PyTorch silently builds against a multi-arch fallback list
    on machines without a GPU, producing bloated binaries that take 10x longer
    to compile and almost never match what the user actually wanted.
    """
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        "Cannot compile fused-ssim CUDA extension: no local GPU detected and "
        "TORCH_CUDA_ARCH_LIST is not set.\n"
        "Either:\n"
        "  - Set TORCH_CUDA_ARCH_LIST explicitly: "
        "TORCH_CUDA_ARCH_LIST='8.9' pip install .\n"
        "  - Set BUILD_NO_CUDA=1 to skip extension compilation entirely.\n"
        "  - Run on a machine with a CUDA-capable GPU (PyTorch will "
        "auto-detect its compute capability)."
    )


def configure_cuda():
    log("[fused-ssim] Compiling for CUDA.")
    nvcc_flags = ["-O3", "--maxrregcount=32"]
    if not NO_FAST_MATH:
        nvcc_flags.append("--use_fast_math")
    if LINE_INFO:
        nvcc_flags.append("-lineinfo")
    if NVCC_FLAGS_EXTRA:
        nvcc_flags.extend(NVCC_FLAGS_EXTRA.split())

    cxx_flags = ["-O3"]

    if torch.version.hip:
        log("[fused-ssim] Detected AMD GPU with ROCm/HIP.")
        nvcc_flags.append("-ffast-math")
        detected = "AMD GPU (ROCm/HIP)"
    else:
        _fail_if_no_gpu_and_no_arch_list()
        detected = os.environ.get("TORCH_CUDA_ARCH_LIST") or "auto-detect from local GPU"

    return CUDAExtension, "ssim.cu", "fused_ssim_cuda", {"cxx": cxx_flags, "nvcc": nvcc_flags}, [], detected


def configure_mps():
    log("[fused-ssim] Compiling for MPS.")
    return (
        CppExtension,
        "ssim.mm",
        "fused_ssim_mps",
        {"cxx": ["-O3", "-std=c++17", "-ObjC++", "-Wno-unused-parameter"]},
        ["-framework", "Metal", "-framework", "Foundation"],
        "Apple Silicon (MPS)",
    )


def configure_xpu():
    log("[fused-ssim] Compiling for XPU.")
    os.environ.setdefault("CXX", "icpx")
    try:
        device_name = torch.xpu.get_device_name(0)
        detected = f"Intel XPU (SYCL) - {device_name}"
    except Exception:
        detected = "Intel XPU (SYCL)"
    return (
        CppExtension,
        "ssim_sycl.cpp",
        "fused_ssim_xpu",
        {"cxx": ["-O3", "-std=c++17", "-fsycl"]},
        ["-fsycl"],
        detected,
    )


def detect_backend():
    if torch.cuda.is_available() or torch.version.hip:
        return configure_cuda()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return configure_mps()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return configure_xpu()
    # No backend detected at runtime; fall through to CUDA which will either
    # honor TORCH_CUDA_ARCH_LIST or raise via _fail_if_no_gpu_and_no_arch_list.
    return configure_cuda()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
if BUILD_NO_CUDA:
    log("[fused-ssim] BUILD_NO_CUDA=1: skipping extension compilation.")
    setup(name="fused_ssim", packages=["fused_ssim"], ext_modules=[], cmdclass={})
else:
    extension_type, extension_file, build_name, compiler_args, link_args, detected_arch = detect_backend()

    extra_link_args = list(link_args)
    if not WITH_SYMBOLS and "xpu" not in build_name and not torch.backends.mps.is_available():
        extra_link_args.append("-s")

    class CustomBuildExtension(BuildExtension):
        def build_extensions(self):
            if "xpu" in build_name:
                self.compiler.compiler_so = ["icpx"] + self.compiler.compiler_so[1:]
                self.compiler.compiler_cxx = ["icpx"] + self.compiler.compiler_cxx[1:]
                self.compiler.linker_so = ["icpx"] + self.compiler.linker_so[1:]
            log("\n" + "=" * 60)
            log(f"[fused-ssim] backend: {detected_arch}")
            log(f"[fused-ssim] TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST', '(auto-detect from local GPU)')}")
            log(f"[fused-ssim] MAX_JOBS: {os.environ.get('MAX_JOBS')}")
            log("=" * 60 + "\n")
            super().build_extensions()

    setup(
        name="fused_ssim",
        packages=["fused_ssim"],
        ext_modules=[
            extension_type(
                name=build_name,
                sources=[extension_file, "ext.cpp"],
                extra_compile_args=compiler_args,
                extra_link_args=extra_link_args,
            )
        ],
        cmdclass={"build_ext": CustomBuildExtension.with_options(verbose=VERBOSE)},
    )

    log(f"[fused-ssim] Setup complete. backend={detected_arch}")

if _need_to_unset_max_jobs:
    os.environ.pop("MAX_JOBS", None)
