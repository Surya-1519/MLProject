from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List:
    #Function to get requirements
    requirements = []
    with open(file_path) as file_obj:
        requirements = [req.replace("\n","") for req in file_obj.readlines()]
        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements

setup(
    name="ML_Project",
    version='0.0.1',
    author="Surya",
    author_email="surya19teja.sripati@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")
)
