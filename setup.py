from setuptools import setup, find_packages

setup(
    name="attack_rag",
    version="0.1.0",
    description="RAG Attack Framework",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
)