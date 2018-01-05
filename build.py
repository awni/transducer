import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.abspath(__file__)

sources = ['src/transducer.c']
headers = ['src/transducer.h']

ffi = create_extension(
    '_ext.transducer',
    headers=headers,
    sources=sources,
    relative_to=__file__,
    extra_compile_args=["-std=c99", "-fopenmp"]
)

if __name__ == '__main__':
    ffi.build()
