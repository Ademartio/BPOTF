# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, build_ext
from setuptools import setup
import sys
import os
import glob

__version__ = "0.0.1"

if os.path.exists("./build") is not True:
    os.mkdir("./build")

if sys.platform == "win32":
    compile_flags = ["/Ox", "/std:c++20"]
    std = ""
    extra_link_args =[]
else:
    # Set environment compiler, pybind uses CCompiler from distutils
    # so thats why CC is set to g++.
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"
    std = "2a"
    #os.environ["CFLAGS"] = "-O3"
    os.environ["CPPFLAGS"] = "-O3"
    extra_link_args =[]

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

# Get and sort source files for reproducibility
sources = sorted(glob.glob("src/**/*.cpp", recursive=True))
for elem in sources:
    if "main_test.cpp" in elem:
        sources.remove(elem)

includes = ["src"]

# Trying to change default installing library...
# Still no success, to-do
# OUTDIR = "./module"
# if os.path.exists(OUTDIR) is not True:
#     os.mkdir(OUTDIR)

ext_modules = [
    Pybind11Extension(
        "BPOTF",
        sources,
        cxx_std=std,
        include_dirs=includes,
        language="c++"
    )
]

setup(
    name="BPOTF",
    version=__version__,
    author="Imanol Etxezarreta",
    author_email="ietxezarretam@gmail.com",
    url="https://github.com/Ademartio/BPBP/tree/OBPOTF",
    description="Implementation of Belief Propagation Ordered Tanner Forest decoding method.",
    long_description="",
    ext_modules=ext_modules,
    #extras_require={},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
