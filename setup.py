from setuptools import setup, find_packages

setup(
    name="crystalmind",
    version="1.0.0",
    description="Advanced XRD Crystal Structure Classification",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "mp-api>=0.41.0",
        "pymatgen>=2024.1.1",
        "mlflow>=2.9.0",
        "streamlit>=1.30.0",
        "plotly>=5.18.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
    ],
)
