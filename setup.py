"""
Setup script for XeroGraph.
XeroGraph  - Drawing from the Greek root "xero" for dry or void, and "graph" for writing or recording,
the name conveys the idea of filling in or examining the 'dry' spots (missing values) in data.
"""
import sys

# Check if setuptools is installed
try:
    from setuptools import setup, find_packages
except ImportError:
    print("setuptools is required to install XeroGraph. Please install it first.")
    sys.exit(1)

# Check Python version
if not (sys.version_info[0] == 3 and sys.version_info[1] >= 9):
    sys.exit(
        f'XeroGraph requires Python 3.9 or higher. '
        f'You are using Python {sys.version_info[0]}.{sys.version_info[1]}.'
    )

# Read long description from README.md
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ("XeroGraph is developed for research purposes only to perform Little's MCAR test."
                        "It also visualizes missing values.")

# Main setup configuration
setup(
    name='XeroGraph',
    version='0.0.2',
    author='Julhash Kazi',
    author_email='XeroGraph@kazilab.se',
    url='https://www.kazilab.se',
    description="A Python implementation of Little's MCAR test and missing value visualization",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    install_requires=[
        'pandas>=2.1',
        'numpy>=1.24',
        'scipy>=1.11',
        'matplotlib>=3.8',
        'statsmodels>=0.14',
        'scikit-learn>=1.4',
        'xgboost>=2.1'
    ],
    platforms='any',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires='>=3.9',  # Ensure this matches your compatibility checks
)
