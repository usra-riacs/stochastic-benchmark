import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stochastic-benchmark",
    version="0.0.1",
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
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
