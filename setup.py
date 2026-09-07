from setuptools import setup, find_packages

setup(
    name="cepler",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "scikit-learn",
        "pynvml",
    ],
    author="Your Name",
    description="Ray Attention Transformer for energy-efficient inference",
    license="MIT",
    url="https://github.com/mikky1sCp/Cepler-23-official",
    python_requires=">=3.8",
)