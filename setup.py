from setuptools import setup, find_packages

setup(
    name="eeg_viewer",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "mne>=1.6.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
        "fooof>=1.0.0",
    ],
    python_requires=">=3.9",
)
