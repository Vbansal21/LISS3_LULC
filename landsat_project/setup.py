from setuptools import setup, find_packages

setup(
    name="landsat_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.2",
        "rasterio>=1.2.0",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.3.4",
        "tqdm>=4.62.3",
        "pillow>=8.3.1",
        "opencv-python>=4.5.3",
        "tensorboard>=2.7.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Land Use Land Cover Classification and Change Detection using Landsat data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/landsat_project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 