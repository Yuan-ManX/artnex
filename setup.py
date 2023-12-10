# MIT License

# Copyright (c) 2023 Yuan-Man

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Install artnex."""

__doc__ = """
ArtNex: ArtNex is a deep learning framework exploring the innovative fusion of art and technology.
See the README file for details, usage info, and a list of feature.
"""

import os
import sys
import setuptools

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'artnex')
sys.path.append(version_path)
from artnex import __version__  

setuptools.setup(
    name='artnex',
    version=__version__,
    description='ArtNex is a deep learning framework exploring the innovative fusion of art and technology.',
    author='Yuan Man',
    author_email='ym1076302261@gmail.com',
    url='https://github.com/Yuan-ManX/artnex',
    license='MIT',
    packages=setuptools.find_packages(),
    long_description=__doc__,
    install_requires=[
        'numpy',
        'contextlib',
        'cupy',
        'matplotlib',
        'urllib',
        'requests',
        'logging',
        'onnx',
        'onnxruntime',
        'PIL',
        'librosa',
        'collections',
        'tqdm',
        'gzip',
        'pickle',
        'tqdm',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='machinelearning deeplearning artificialintelligence neuralnetworks dataprocessing art-techconvergence',
)