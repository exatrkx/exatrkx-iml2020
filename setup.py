from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

description="Deep learning based tracking reconstruction"

setup(
    name="exatrkx",
    version="1.0.0",
    description="Library for building tracks with Graph Neural Networks.",
    long_description=description,
    author="Exa.TrkX Collaboration",
    author_email="",
    license="Apache License, Version 2.0",
    keywords=["graph networks", "track formation", "tracking", "machine learning"],
    url="https://github.com/exatrkx/exatrkx-iml2020",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        #"tensorflow >= 2.2",
        "torch", "pytorch-lightning",
        "faiss-gpu"
        "graph_nets>=1.1",
        "future",
        "networkx>=2.4",
        "scipy",
        "pandas",
        "setuptools",
        "matplotlib",
        'sklearn',
        'pyyaml>=5.1',
        'trackml@ https://github.com/LAL/trackml-library/tarball/master#egg=trackml-3',
        'tables',
        'more-itertools',
    ],
    package_data = {
        "exatrkx": ["config/*.yaml"]
    },
    extras_require={
    },
    setup_requires=['trackml'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    scripts=[

    ],
)