from setuptools import setup, find_packages

setup(
    name="mcp_rag",
    version="0.1.0",
    description="Sistema RAG con arquitectura Model-Context-Protocol",
    author="Manuel Benitez Sanchez",
    author_email="Manuel.BENITEZ-SANCHEZ@ext.ec.europa.eu",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.12",
    install_requires=[
        "requests>=2.32.2",
        "python-dotenv>=1.0.0",
        "numpy>=1.26.4",
        "openai>=1.51.2",
        "qdrant-client>=1.13.2",
        "fastapi>=0.115.11",
        "uvicorn>=0.34.0",
        "pydantic>=2.5.3",
        "pytest>=7.4.4"
    ],
)