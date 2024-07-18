# Contributing to XeroGraph

## Welcome!
We're excited that you're interested in contributing to XeroGraph! This document provides guidelines and information about contributing to this project. We hope to make the process as transparent and friendly as possible. Your contributions are essential for keeping XeroGraph great.

## Getting Started
Before you start, you need to have Python installed along with the pandas, numpy, and matplotlib libraries. Familiarity with Git for version control is also recommended.

### Set Up Your Development Environment
1. Fork the XeroGraph repository.
2. Clone your fork to your local machine:
```bash
git clone https://github.com/your-username/xerograph.git
```
3. Set up a virtual environment in the project directory:
```bash
python -m venv env
```
#### On Linux/Mac
```bash
source env/bin/activate
```
#### On Windows
```bash
env\Scripts\activate  
```
4. Install the development dependencies:
```bash
pip install -r requirements_dev.txt
```
## Making Changes
1. Create a new branch for your changes:
```bash
git checkout -b name-of-your-branch
```
2. Make your changes locally.
3. Write or adapt tests as necessary.
4. Run the tests to ensure they pass.
5. Commit your changes:
```bash
git commit -am "Add a brief description of your change"
```
5. Push your changes to your fork:
```bash
git push origin name-of-your-branch
```

## Submitting a Pull Request
1. Go to the repository on GitHub.
2. Select your fork.
3. Press the 'Pull Request' button.
4. Base the pull request against the main project's main branch.
5. Describe the changes and why you made them.
6. Submit the pull request.

## Code Review Process
Each pull request must be reviewed and approved by at least one of the project maintainers. We aim to review all pull requests within a week. If there are any comments or adjustments you need to make, we will let you know.

## Community
For any questions or major changes you are considering, please file an issue first to discuss what you would like to change. Join our community chat on [platform] to get more involved.

## Code of Conduct
Contributors are expected to uphold the Code of Conduct, which promotes a welcoming and inclusive environment.

## Thank You
Thank you for contributing to XeroGraph! Every contribution is important and helps make this project what it is.