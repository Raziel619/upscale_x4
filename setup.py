from pathlib import Path

from setuptools import find_packages, setup

# read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="upscale_x4",
    version="0.0.1",
    license="MIT",
    author="Raziel619",
    author_email="raziel619dev@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/Raziel619/upscale_x4",
    keywords="upscale",
    install_requires=[],
)
