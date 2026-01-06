from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension
import torch
import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stderr.reconfigure(line_buffering=True)


def log(msg):
    """Print to both stdout and stderr."""
    print(msg)
    print(msg, file=sys.stderr, flush=True)


def configure_cuda():
    """Configure CUDA/ROCm backend."""
    log("Compiling for CUDA.")
    compiler_args = {"cxx": ["-O3"], "nvcc": ["-O3"]}

    if torch.version.hip:
        log("Detected AMD GPU with ROCm/HIP")
        compiler_args["nvcc"].append("-ffast-math")
        detected_arch = "AMD GPU (ROCm/HIP)"
    else:
        compiler_args["nvcc"].extend(("--maxrregcount=32", "--use_fast_math"))

        # If the user set TORCH_CUDA_ARCH_LIST, PyTorch will honor it.
        # If not set, we set a conservative default list.
        if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
            # "Common" modern archs; adjust if you want broader/leaner.
            os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.9;9.0;12.0"
            log(f"TORCH_CUDA_ARCH_LIST not set; defaulting to {os.environ['TORCH_CUDA_ARCH_LIST']}")
        detected_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")

    return CUDAExtension, "ssim.cu", "fused_ssim_cuda", compiler_args, [], detected_arch


def configure_mps():
    """Configure Apple MPS backend."""
    log("Compiling for MPS.")
    compiler_args = {"cxx": ["-O3", "-std=c++17", "-ObjC++", "-Wno-unused-parameter"]}
    link_args = ["-framework", "Metal", "-framework", "Foundation"]
    return CppExtension, "ssim.mm", "fused_ssim_mps", compiler_args, link_args, "Apple Silicon (MPS)"


def configure_xpu():
    """Configure Intel XPU (SYCL) backend."""
    log("Compiling for XPU.")
    os.environ['CXX'] = 'icpx'

    compiler_args = {"cxx": ["-O3", "-std=c++17", "-fsycl"]}
    link_args = ["-fsycl"]

    try:
        device_name = torch.xpu.get_device_name(0)
        log(f"Detected Intel XPU: {device_name}")
        detected_arch = f"Intel XPU (SYCL) - {device_name}"
    except Exception:
        log("Detected Intel XPU (SYCL)")
        detected_arch = "Intel XPU (SYCL)"

    return CppExtension, "ssim_sycl.cpp", "fused_ssim_xpu", compiler_args, link_args, detected_arch


# Detect backend
if torch.cuda.is_available():
    extension_type, extension_file, build_name, compiler_args, link_args, detected_arch = configure_cuda()
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    extension_type, extension_file, build_name, compiler_args, link_args, detected_arch = configure_mps()
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    extension_type, extension_file, build_name, compiler_args, link_args, detected_arch = configure_xpu()
else:
    extension_type, extension_file, build_name, compiler_args, link_args, detected_arch = configure_cuda()

# Create a custom class that prints the architecture information
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        # For SYCL, override compiler to use icpx
        if 'xpu' in build_name:
            self.compiler.compiler_so = ['icpx'] + self.compiler.compiler_so[1:]
            self.compiler.compiler_cxx = ['icpx'] + self.compiler.compiler_cxx[1:]
            self.compiler.linker_so = ['icpx'] + self.compiler.linker_so[1:]

        arch_info = f"Building with TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST','(not set)')}"
        print("\n" + "="*50)
        print(arch_info)
        print("="*50 + "\n")
        super().build_extensions()

setup(
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        extension_type(
            name=build_name,
            sources=[
                extension_file,
                "ext.cpp"],
            extra_compile_args=compiler_args,
            extra_link_args=link_args
        )
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    }
)

# Print again at the end of setup.py execution
if "nvcc" in compiler_args:
    final_msg = "Setup completed. NVCC args: {}. CXX args: {}. Link args: {}.".format(
        compiler_args["nvcc"], compiler_args["cxx"], link_args
    )
else:
    final_msg = "Setup completed. CXX args: {}. Link args: {}.".format(
        compiler_args["cxx"], link_args
    )
print(final_msg)
