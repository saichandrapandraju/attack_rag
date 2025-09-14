from setuptools import setup, find_packages

setup(
    name="attack_rag",
    version="1.0.0",
    description="PoisonedRAG Core: Adversarial Document Generation for RAG Systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Sai Chandra Pandraju",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "openai>=1.0.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ]
    },
    keywords="adversarial-attacks rag retrieval-augmented-generation nlp security",
)