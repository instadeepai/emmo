"""Setup code for EMMo, in particular, for defining the C extension(s)."""
import numpy as np
from setuptools import Extension
from setuptools import setup

extension_mhc1_deconv = Extension(
    "emmo.em.c_extensions.mhc1_c_ext",
    sources=["emmo/em/c_extensions/mhc1.c", "emmo/em/c_extensions/utils.c"],
    include_dirs=[np.get_include()],
)

extension_mhc2_deconv = Extension(
    "emmo.em.c_extensions.mhc2_c_ext",
    sources=["emmo/em/c_extensions/mhc2.c", "emmo/em/c_extensions/utils.c"],
    include_dirs=[np.get_include()],
)

setup(name="emmo", ext_modules=[extension_mhc1_deconv, extension_mhc2_deconv])
