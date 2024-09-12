"""Setup code for EMMo, in particular, for defining the C extension(s)."""
import numpy as np
from setuptools import Extension
from setuptools import setup

extension_mod = Extension(
    "emmo.em.mhc1_c_ext",
    sources=["emmo/em/mhc1.c"],
    include_dirs=[np.get_include()],
)

setup(name="emmo", ext_modules=[extension_mod])
