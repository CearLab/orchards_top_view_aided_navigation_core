# from distutils.core import setup
# from Cython.Build import cythonize
#
# setup(
#     ext_modules = cythonize("contours_scan.pyx")
# )


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension('contours_scan', ['src/contours_scan.pyx'])
]

setup(
    ext_modules = cythonize(extensions)
)