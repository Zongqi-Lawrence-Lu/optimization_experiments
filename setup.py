from setuptools import setup, find_packages

setup(
    name="optimization_framework",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "pyyaml",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
    ],
    extras_require={
        "nlp": ["transformers", "datasets", "sacrebleu", "nltk"],
        "vision": ["torchvision"],
        "dev": ["pytest"],
    },
)
