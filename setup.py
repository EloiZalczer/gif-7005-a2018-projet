import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gif-7005-a2018-projet",
    version="0.0.1",
    author="Arnaud Dalie, Rosalie Kletzander, Eloi Zalczer",
    description="UMANX sound recognition project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EloiZalczer/gif-7005-a2018-projet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: Ubuntu 16.04",
    ],
)
