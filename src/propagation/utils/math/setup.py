#To build the cython code in the .pyx file, type in the terminal:
#"python setup.py build_ext --inplace"
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# extensions = [
#     Extension("differential_operators", ["differential_operators.pyx"],
#               include_dirs=[numpy.get_include()]),
# ]

# setup(
#     name="my_cython_fd",
#     ext_modules=cythonize(extensions, annotate=True),
# )

setup(
    name="cython_differential_operators",
    ext_modules=cythonize(["cython_differential_operators.pyx"]),
    include_dirs=[numpy.get_include()],
)
