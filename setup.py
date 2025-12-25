from setuptools import setup, find_packages

setup(
    name="sparsity-trap",
    version="1.0.0",
    author="Max Van Belkum",
    author_email="max.vanbelkum@vanderbilt.edu",
    description="The Sparsity Trap: Why MSE Fails and Poisson Succeeds for 2um Spatial Transcriptomics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vanbelkummax/mse-vs-poisson-2um-benchmark",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-image>=0.20.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pillow>=9.5.0",
        "pytest>=7.3.0",
        "pytest-cov>=4.1.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
