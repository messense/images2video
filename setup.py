#!/usr/bin/env python
from __future__ import with_statement
import os
from setuptools import setup, find_packages

readme = 'README.md'
if os.path.exists('README.rst'):
    readme = 'README.rst'
with open(readme) as f:
    long_description = f.read()

setup(
    name='images2video',
    version='0.1',
    author='messense',
    author_email='messense@icloud.com',
    url='https://github.com/messense/images2video',
    packages=find_packages(),
    keywords='opencv, video',
    description='Python images to video library using OpenCV',
    long_description=long_description,
    install_requires=['numpy'],
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
    ],
)
