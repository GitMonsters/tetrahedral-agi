from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tetrahedral-agi",
    version="1.0.0",
    author="Tetrahedral AI Team",
    author_email="contact@tetrahedral-ai.com",
    description="64-Point Tetrahedron AI Framework for Geometric Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tetrahedral-ai/tetrahedral-agi",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "tqdm>=4.62.0",
        "requests>=2.25.0",
        "websockets>=10.0",
    ],
    extras_require={
        "training": [
            "wandb>=0.12.0",
            "optuna>=3.0.0",
        ],
        "scientific": [
            "biopython>=1.79",
            "pymatgen>=2022.0.0",
            "ase>=3.22.0",
        ],
        "manufacturing": [
            "opencv-python>=4.5.0",
            "scikit-learn>=1.0.0",
            "open3d>=0.13.0",
        ],
        "autonomous": [
            "scipy>=1.7.0",
            "transforms3d>=0.3.1",
        ],
        "all": [
            "wandb>=0.12.0",
            "optuna>=3.0.0",
            "biopython>=1.79",
            "pymatgen>=2022.0.0",
            "ase>=3.22.0",
            "opencv-python>=4.5.0",
            "scikit-learn>=1.0.0",
            "open3d>=0.13.0",
            "transforms3d>=0.3.1",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tetrahedral-agi=tetrahedral_agi.cli:main",
            "tetrahedral-api=tetrahedral_agi.api.api_gateway:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tetrahedral_agi": [
            "configs/*.yaml",
            "data/*.json",
        ],
    },
)