from setuptools import setup, find_packages

setup(
    name="sympauli",
    version="0.1.0",
    description="Symbolic Pauli Heisenberg Evolution Engine",
    packages=find_packages(exclude=["tests*"]),
    install_requires=["sympy>=1.12", "numpy>=1.24"],
    extras_require={"dev": ["pytest>=7"]},
    python_requires=">=3.10",
)
