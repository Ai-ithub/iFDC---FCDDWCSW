from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="drilling_data_processor",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced drilling data preprocessing toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/drilling-data-processor",
    
    packages=find_packages(where="drilling_data_processor"),

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    
    python_requires=">=3.8",
    
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "pyarrow>=6.0.0",
        "tqdm>=4.0.0"
    ],
    
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0"
        ],
        "visualization": [
            "matplotlib>=3.0",
            "seaborn>=0.11.0"
        ]
    },
    
    entry_points={
        "console_scripts": [
            "drilling-process=drilling_data_processor.drilling_processor.cli:main",
        ],
    },

    include_package_data=True,
    
    package_data={
        "drilling_data_processor.drilling_processor": ["config/*.json"]
    }
)
