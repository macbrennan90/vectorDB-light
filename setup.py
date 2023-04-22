from setuptools import setup, find_packages

setup(
    name="vectorDB_light",
    version="0.1.0",
    packages=find_packages(where="src", exclude=["tests"]),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scikit-learn",
    ],
)
