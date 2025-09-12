import os
import shutil

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import Extension, find_packages, setup

# Define the path to the Cython source files
cython_path = os.path.join("tierpsynn", "extras", "cython_files")
spline_cython_pyx = os.path.join(cython_path, "spline_cython.pyx")


# Custom build_ext command to move .so files to the desired location
class CustomBuildExt(build_ext):
    def build_extension(self, ext):
        # Call the original build_extension method
        super().build_extension(ext)
        # Get the full path of the built .so file
        built_path = self.get_ext_fullpath(ext.name)
        # Define the target directory where the .so file should be moved
        target_dir = os.path.abspath(cython_path)
        # Move the built .so file to the target directory
        shutil.move(built_path, os.path.join(target_dir, os.path.basename(built_path)))


cuda_compile_args = [
    "-arch=sm_90",  # Replace XX with your GPU's compute capability (e.g., sm_75)
    "-rdc=true",  # Relocatable device code
    "-O3",
    "--std=c++17",  # Or your desired C++ standard
]
cuda_link_args = [
    "-lcudart",  # Link against CUDA runtime library
]


extensions = [
    Extension(
        "spline_cython",
        [spline_cython_pyx],
        language="c++",  # Important for C++ and CUDA
        extra_compile_args=cuda_compile_args,
        extra_link_args=cuda_link_args,
        # Include directories for CUDA headers if needed
        include_dirs=["/usr/local/cuda/include", np.get_include()],
        # Library directories for CUDA libraries if needed
        library_dirs=["/usr/local/cuda/lib64"],
    )
]


# Configuration for the setup
setup(
    packages=find_packages(),
    zip_safe=False,
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},  # Ensure Python 3 language level
    ),
    cmdclass={"build_ext": CustomBuildExt},
)
