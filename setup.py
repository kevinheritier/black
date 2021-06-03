import setuptools
from setuptools import setup


setup(
    name='blacklitterman',
    version='0.1.0',
    author='Kevin Heritier, Dorian Deneuville',
    author_email='kevheritier@gmail.com, dorian.deneuville@gmail.com',
    packages=setuptools.find_packages(),
    url="https://github.com/kevinheritier/black",
    description='Set of tools for portfolio optimization using Black-Litterman models',
    install_requires=[
        "dash >= 1.14.0",
        "numpy >= 1.19.1",
        "pandas >= 1.1.0",
        "scipy >= 1.5.2"
    ],
    python_requires='>=3.6',
)
