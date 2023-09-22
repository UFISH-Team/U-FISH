from setuptools import setup, find_packages
import re


classifiers = [
    "Development Status :: 3 - Alpha",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Science/Research",
]


keywords = [
    'Spot detection', 'FISH', 'Bioimage analysis',
    'Image processing', 'Deep learning', 'U-Net',
]


URL = "https://github.com/GangCaoLab/U-FISH"


def get_version():
    with open("ufish/__init__.py") as f:
        for line in f.readlines():
            m = re.match("__version__ = '([^']+)'", line)
            if m:
                return m.group(1)
        raise IOError("Version information can not found.")


def get_long_description():
    return f"See {URL}"


def get_requirements_from_file(filename):
    requirements = []
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements


def get_install_requires():
    return get_requirements_from_file('requirements.txt')


def get_doc_requires():
    return get_requirements_from_file('docs/requirements.txt')


requires_test = ['pytest', 'pytest-cov', 'pytest-asyncio', 'flake8', 'mypy']
packages_for_dev = ["pip", "setuptools", "wheel", "twine", "ipdb"]

requires_dev = packages_for_dev + requires_test + get_doc_requires()


setup(
    name='ufish',
    author='Weize Xu, Huaiyuan Cai, Qian Zhang',
    author_email='vet.xwz@gmail.com',
    version=get_version(),
    license='MIT',
    description='Deep learning based spot detection for FISH images.',
    long_description=get_long_description(),
    keywords=keywords,
    url=URL,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    classifiers=classifiers,
    install_requires=get_install_requires(),
    extras_require={
        'dev': requires_dev,
        'onnxruntime-gpu': ['onnxruntime-gpu'],
    },
    python_requires='>=3.9, <4',
    entry_points={
        'console_scripts': [
            'ufish = ufish.__main__:main',
        ],
    },
)
