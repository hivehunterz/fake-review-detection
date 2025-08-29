from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fake-review-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced Fake Review Detection System using BART and Multi-LLM Labeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fake-review-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "label-reviews=google_reviews_labeler:main",
            "train-bart=enhanced_bart_finetune:main",
            "evaluate-bart=evaluate_fine_tuned_bart:main",
        ],
    },
    keywords="fake review detection, NLP, BART, machine learning, text classification",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/fake-review-detection/issues",
        "Source": "https://github.com/yourusername/fake-review-detection",
        "Documentation": "https://github.com/yourusername/fake-review-detection/blob/main/README.md",
    },
)
