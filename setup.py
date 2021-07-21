import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ilurl",
    version="0.0.1.cityflow",
    author="Guilherme Varela",
    author_email="guilhermevarela@protonmail.com",
    description=
    "iLU: Integrative Learning from Urban Data and Situational Context for City Mobility Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GAIPS/ILU-RL",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
