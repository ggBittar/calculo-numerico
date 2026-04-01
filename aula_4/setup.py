from __future__ import annotations

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


extensions = [
    Extension(
        name="app.backend.pde_kernels",
        sources=[str(Path("app") / "backend" / "pde_kernels.pyx")],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
