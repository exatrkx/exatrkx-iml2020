from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

description="Deep learning based tracking reconstruction"

setup(
    name="exatrkx",
    version="1.2.0",
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
        "pytorch-lightning==1.0.2",
        "faiss-gpu",
        'trackml@ https://github.com/LAL/trackml-library/tarball/master#egg=trackml-3',
        "graph_nets>=1.1",
        # "tensorflow",
        # 'horovod',
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
        'exatrkx/scripts/run_lightning.py',
        'exatrkx/scripts/convert2tf.py',
        'exatrkx/scripts/train_gnn_tf.py',
        'exatrkx/scripts/eval_gnn_tf.py',
        'exatrkx/scripts/tracks_from_gnn.py',
        'exatrkx/scripts/eval_reco_trkx.py',
        'exatrkx/scripts/install_geometric.sh',
    ],
)