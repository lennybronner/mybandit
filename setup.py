from setuptools import setup, find_packages

setup(
    name="bandits",
    version="0.1",
    description="A collection of bandit algorithm implementations for experimentation",
    author="Your Name",
    packages=find_packages(include=["bandits", "bandits.*"]),
    python_requires=">=3.13",
    install_requires=[
        "numpy>=2.2.0"
    ],
)