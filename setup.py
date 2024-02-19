from setuptools import setup, find_packages

setup(
    name="segments_runner",
    version="0.1",
    description="Run multiple segments of EdgeTPU models in sequence",
    author="Changhun Han",
    author_email="ehwjs1914@ajou.ac.kr",
    packages=find_packages(exclude=["test"]),
)
