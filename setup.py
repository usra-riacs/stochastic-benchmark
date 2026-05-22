import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
    "cloudpickle>=2.2",
    "dill>=0.3.5",
    "fonttools>=4.25",
    "matplotlib>=3.7",
    "mkl-service>=2.5.2",
    "multiprocess>=0.70.18",
    "munkres>=1.1.4",
    "networkx>=3.0",
    "numpy>=2.0",
    "pandas>=2.3",
    "scipy>=1.11",
    "seaborn>=0.13.2",
    "tqdm>=4.66",
]

EXTRAS_REQUIRE = {
    "examples": [
        "scikit-learn>=1.3.0",
        "dimod>=0.12",
    ],
    "notebooks": [
        "nbconvert>=7",
        "ipykernel>=6",
    ],
    "generation": [
        "hyperopt>=0.2.7",
        "setuptools<81",
    ],
}

setuptools.setup(
    name="stochastic-benchmark",
    version="0.1.0",
    author="David Bernal",
    author_email="dbernalneira@usra.edu",
    description="A package to analyze benchmarking results of stochastic optimization solvers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bernalde/stochastic-benchmark",
    project_urls={
        "Bug Tracker": "https://github.com/bernalde/stochastic-benchmark/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
