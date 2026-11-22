"""Check which calculation kernels were compiled into the package.

The upstream tests pass with any set of kernels, so they do not notice a
build that silently lost its SIMD or CUDA kernels.

usage: python check_kernels.py [--cuda]
"""

import sys

from diffpy.srreal.calculationkernel import (
    availableCalculationKernels,
    setCalculationKernel,
)

CUDA_KERNELS = ("cuda", "cuda-mixed")

names = availableCalculationKernels()
print("available calculation kernels:", names)

# Without xsimd the build falls back to the scalar kernel alone.  Every
# x86_64 CPU has SSE2 and every arm64 CPU has NEON.
simd = [n for n in names if n != "scalar" and n not in CUDA_KERNELS]
assert simd, f"no SIMD kernels were built, got {names}"

if "--cuda" in sys.argv[1:]:
    if "cuda" in names:
        print("CUDA kernels are available on this machine")
    else:
        # A machine without a CUDA device does not list the CUDA kernels.
        # The error of selecting one tells that from a build without CUDA.
        try:
            setCalculationKernel("cuda")
        except ValueError as e:
            print("CUDA kernels are built, but not usable here:", e)
            assert "built without CUDA" not in str(e), e
        else:
            raise AssertionError("an unlisted CUDA kernel was selected")
