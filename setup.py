#!/usr/bin/env python3

import os
import re
import subprocess
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = "0.0.0"


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cmake_version = re.search(r"version\s*([\d.]+)", out.decode().lower()).group(1)
        cmake_version = [int(i) for i in cmake_version.split(".")]
        if cmake_version < [3, 10, 0]:
            raise RuntimeError("CMake >= 3.10.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        srcdir = os.path.abspath("transducer")
        # required for auto - detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DPROJECT_SOURCE_DIR=" + srcdir,
            "-DTRANSDUCER_BUILD_PYTHON_BINDINGS=ON",
            "-DTRANSDUCER_BUILD_TESTS=OFF",
            "-DCMAKE_BUILD_TYPE=" + cfg,
        ]

        build_args = ["--config", cfg, "--", "-j4"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", srcdir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


setup(
    name="transducer",
    version=__version__,
    author="Awni Hannun",
    description="Fast RNN Transducer",
    url="https://github.com/awni/transducer",
    packages=["transducer"],
    ext_modules=[
        CMakeExtension("transducer._transducer"),
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    license="MIT licensed, as found in the LICENSE file",
    python_requires=">=3.5",
)
