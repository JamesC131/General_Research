import os
import ctypes
import sys

# Load Soar native library FIRST
ctypes.CDLL("/home/james/Downloads/Soar-releases-9.6.4/out/libSoar.so")

sys.path.append("/home/james/Downloads/Soar-releases-9.6.4/out")
import Python_sml_ClientInterface as sml

_kernel = None

def get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = sml.Kernel.CreateKernelInCurrentThread(True, 0)
    return _kernel
