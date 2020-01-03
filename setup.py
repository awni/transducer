from setuptools import setup, Extension
import sys
from torch.utils import cpp_extension

if sys.platform == "darwin":
    args = ["-DAPPLE"]
else:
    args = ["-fopenmp"]


ext = Extension(
        name='transducer_cpp',
        sources=['transducer.cpp'],
        include_dirs=cpp_extension.include_paths(),
        extra_compile_args=args,
        language='c++')

setup(name='transducer_cpp',
      ext_modules=[ext],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

