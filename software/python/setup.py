import setuptools

with open("README.md") as fh:
    long_description = fh.read()

version = "0.1.4a"

setuptools.setup(
    name="nextwheel",
    version=version,
    description="Python module for NextWheel instrumented wheels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/felixchenier/NextWheel",
    author="NextWheel collaborators",
    author_email="chenier.felix@uqam.ca",
    license="Apache",
    license_files=["LICENSE.txt", "NOTICE.txt"],
    packages=setuptools.find_packages(),
    package_data={
        "nextwheel": [],
    },
    project_urls={
        "Documentation": "https://github.com/felixchenier/NextWheel/tree/master/python",
        "Source": "https://github.com/felixchenier/NextWheel",
        "Tracker": "https://github.com/felixchenier/NextWheel/issues",
    },
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10",
)
