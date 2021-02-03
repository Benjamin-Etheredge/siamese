import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="siamese",
    version="0.0.10",
    author="Benjamin Etheredge",
    author_email="",
    description="A small siamese network package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Benjamin-Etheredge/siamese",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)