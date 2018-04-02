import os
import sys
import torch
from torch.utils.ffi import create_extension

this_file = os.path.abspath(__file__)

sources = ['src/transducer.c']
headers = ['src/transducer.h']

args = ["-std=c99"]
if sys.platform == "darwin":
    args += ["-DAPPLE"]
else:
    args += ["-fopenmp"]

ffi = create_extension(
    '_ext.transducer',
    headers=headers,
    sources=sources,
    relative_to=__file__,
    extra_compile_args=args
)

if __name__ == '__main__':
    ffi.build()
