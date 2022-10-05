# Import required functions
from gettext import find
from setuptools import setup, find_packages

# Call the setup function
setup(
    author="Uriah Finkel",
    Description="rtichoke for interactive vizualization of performance metrics",
    name="rtichoke",
    version="0.0.1",
    packages=find_packages(include=["rtichoke","rtichoke.*"]),
    install_requires=['pandas', 'numpy']
)