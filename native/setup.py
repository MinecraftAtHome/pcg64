from setuptools import setup, Extension
import importlib.util
import os

numpy_dir = os.path.dirname(importlib.util.find_spec("numpy").origin)
# numpy 2.0 renamed core to _core
numpy_core_dir = os.path.join(numpy_dir, "_core")
if not os.path.exists(numpy_core_dir):
    numpy_core_dir = os.path.join(numpy_dir, "core")

setup(
    ext_modules=[
        Extension(
            name="native",
            sources=["native.cpp", "pcg64.c", "seed_sequence.cpp"],
            include_dirs=[os.path.join(numpy_core_dir, "include"), "."]
        ),
    ]
)