from setuptools import setup, find_packages

setup(
    name="pfc-dynamics-python",
    version="0.1.0",
    description="Python conversion of MATLAB PFC dynamics analysis code",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0", 
        "matplotlib>=3.3.0",
    ],
    python_requires=">=3.7",
)
