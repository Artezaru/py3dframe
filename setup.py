import os
from setuptools import setup, find_packages

# 0. Read the contents of your README.md file to use as the long description
def read_long_description():
    with open("README.md", "r") as file:
        long_description = file.read()
    return long_description

# 1. Read the contents of your requirements.txt file.
def read_requirements():
    with open("requirements.txt", "r") as file:
        # Remove comments and empty lines
        return [line.strip() for line in file.readlines() if line.strip() and not line.startswith('#')]

# 2. Read the version in the __version__.py file
def read_version():
    version_file = "py3dframe/__version__.py"
    with open(version_file, "r") as file:
        exec(file.read()) 
    return locals()["__version__"]

# 3. Setup the package
setup(
    name="py3dframe",
    version=read_version(), # Update the version as necessary
    author="Artezaru",
    author_email="artezaru.github@proton.me",
    description="",  # Short description of the package
    long_description=read_long_description(), # Update the long description as necessary
    long_description_content_type="text/markdown",  # Format of the long description
    url="https://github.com/Artezaru/py3dframe.git",  # URL to the package's homepage
    packages=find_packages(),  # Automatically find packages in the directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Minimum Python version required
    install_requires=read_requirements(),
    include_package_data=True,  # Include package data as specified in MANIFEST.in
    package_data={
        'py3dframe': ['ressources/*'],  # Include all files in ressources/
    },
    entry_points={
        'console_scripts': [
            'py3dframe = py3dframe.__main__:main',  # Define the py3dframe command to run __main__.main()
        ],
    },
)
