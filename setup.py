import os
import setuptools
from typing import List


def read_version_file(rel_path: str) -> List[str]:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read().splitlines()


def get_version(rel_path: str) -> str:
    for line in read_version_file(rel_path=rel_path):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]


with open("README.rst", "r") as f, open("requirements.txt", "r") as g:
    long_description = f.read()
    required = [line for line in g.read().splitlines()]


package_name = "randd"
setuptools.setup(
    name=package_name,
    version=get_version(os.path.join(package_name, '__init__.py')),
    license="MIT License",
    description="A Python Library for Video Codec Comparison.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/UWIVC/randd",
    author="Wentao Liu, Zhengfang Duanmu",
    author_email="liu.wen.tao90@gmail.com, alexduanmu@gmail.com",
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Video",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="rate-distortion performance, video encoding",  # Optional
    packages=setuptools.find_packages(),  # Required
    python_requires=">=3.10, <4",  # TODO: Check whether it works with other python versions
    install_requires=required,
    extras_require={
        "dev": ["flake8", "pytest"],
    },
    project_urls={
        "Bug Reports": "https://github.com/UWIVC/randd/issues",
        "Source": "https://github.com/UWIVC/randd/",
    },
    package_data={
        "randd": ["*.npz"],
        "docs": ["*.png"],
    },
)
