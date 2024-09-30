import os
import shutil

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import find_packages, setup

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


# Configuration for the setup
setup(
    name="tierpsynn",
    version="0.0.1",
    description="Tool for skeletonising worms",
    url="",
    author="Weheliye Hashi",
    author_email="w.weheliye@ic.ac.uk",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    ext_modules=cythonize(
        [spline_cython_pyx],
        compiler_directives={"language_level": "3"},  # Ensure Python 3 language level
    ),
    include_dirs=[np.get_include()],
    cmdclass={"build_ext": CustomBuildExt},
    entry_points={
        "console_scripts": [
            "tierpsynn_process="
            "tierpsynn.processing.main_process_local:process_main_local",
            "ProcessLocal="
            "tierpsynn.processing.processMultipleFilesFun:process_main_file",
            "train_process=" "tierpsynn.train.train:process_main_train",
        ]
    },
)
