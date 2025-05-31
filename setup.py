from setuptools import setup, find_packages

setup(
    name="emotion_classifier",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "scikit-learn",
        "hazm",
        "numpy",
        "imbalanced-learn",
        "joblib",
    ],
    python_requires=">=3.8",
) 