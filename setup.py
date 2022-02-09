from setuptools import setup, Extension
import sys
from torch.utils.cpp_extension import BuildExtension, CppExtension

if sys.platform == "darwin":
    args = ["-DAPPLE"]
else:
    args = ["-fopenmp"]

args.append("-std=c++14")

ext = CppExtension(
        name='transducer_cpp',
        sources=['transducer.cpp'],
        extra_compile_args=args)

setup(name='transducer_cpp',
      version="0.0",
      ext_modules=[ext],
      cmdclass={'build_ext': BuildExtension})

