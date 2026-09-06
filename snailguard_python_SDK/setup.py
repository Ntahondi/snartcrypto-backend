from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="snailguard",
    version="1.0.0",
    author="SnailGuard AI Team",
    description="Advanced AI-Powered API Protection with Zero False Positives and Nuclear Economic Warfare",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=['snailguard', 'snailguard.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Security",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "flask>=2.0.0",
        "fastapi>=0.68.0",
        "pydantic>=1.8.0",
        "xgboost>=1.5.0",  # For XGBoost models
    ],
    include_package_data=True,
    package_data={
        "snailguard": [
            "data/models/*.pkl", 
            "data/models/cascade_models/*.pkl"
        ],
    },
    entry_points={
        'console_scripts': [
            'snailguard-test=snailguard.utils.helpers:test_installation',
        ],
    },
    keywords="security api protection ai ml threat-detection economic-warfare",
    url="https://github.com/your-org/snailguard",
    project_urls={
        "Documentation": "https://docs.snailguard.ai",
        "Source": "https://github.com/your-org/snailguard",
        "Tracker": "https://github.com/your-org/snailguard/issues",
    },
)