#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import
from __future__ import print_function

import io
from os import path

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


here = path.abspath(path.dirname(__file__))


def read(*names, **kwargs):
    return io.open(
        path.join(here, *names),
        encoding=kwargs.get("encoding", "utf8")
    ).read()


long_description = read("README.md")
requirements = read("requirements.txt").split("\n")
optional_requirements = {}

setup(
    name="jax_fdm",
    version="0.4.1",
    description="Auto-differentiable and hardware-accelerated force density method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arpastrana/jax_fdm",
    author="Rafael Pastrana",
    author_email="arpastrana@princeton.edu",
    license="MIT license",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    keywords=["jax", "automatic-differentiation", "architecture", "form-finding", "structural-design"],
    project_urls={},
    packages=["jax_fdm"],
    package_dir={"": "src"},
    package_data={},
    data_files=[],
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    python_requires=">=3.7",
    extras_require=optional_requirements,
    entry_points={
        "console_scripts": [],
    },
    ext_modules=[],
)
