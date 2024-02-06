# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

In this project, participants are tasked with applying their knowledge to pinpoint credit card customers with a high likelihood of churn. The end result will be a set of Python scripts encapsulating a machine learning project, crafted in adherence to the PEP8 coding standards and embodying the best practices in software engineering—modularity, documentation, and thorough testing. Designed for versatility, the package can be operated both interactively and via the command-line interface (CLI). This endeavor serves as a practical exercise in implementing testing, logging, and optimal coding techniques learned in this course. Furthermore, it presents a real-world challenge commonly encountered by data scientists across various industries: identifying customers at risk of churn and devising strategies for retention.

## Files and data description

- **data/**: This directory contains the dataset utilized in the project. These dataset include the information on credit card customers necessary for analyzing and predicting churn.

- **models/**: This folder holds the trained machine learning models that have been developed to predict customer churn. These models are the output of the analysis and training process.

- **Guide.ipynb**: A Jupyter notebook providing guidance on how to navigate through and utilize the project, including steps for data preprocessing, model training, and evaluation.

- **README.md**: The markdown file containing an overview of the project, including its purpose, structure, setup instructions, and usage guidelines.

- **churn_library.py**: A Python library that encapsulates the core functionality of the project, including data preprocessing, model training, and prediction routines, adhering to best coding practices.

- **churn_notebook.ipynb**: A Jupyter notebook that demonstrates the workflow of the project from data loading, analysis, model training, and evaluation, serving as an interactive guide.

- **churn_script_logging_and_tests.py**: This script contains logging and unit tests for the project, ensuring the code quality and reliability of the functions within the Python package.

- **requirements_py3.6.txt**: A file listing the project dependencies required for Python 3.6 environment, ensuring reproducibility.

- **requirements_py3.8.txt**: Similarly, this file specifies the dependencies for the project in a Python 3.8 environment, facilitating the setup in different Python versions.

## Running Files

To run the files in this project and fully utilize its capabilities, follow the steps outlined below for different components:

### 1. Setup Environment

First, ensure that your Python environment is correctly set up with the necessary dependencies:

- Install the required packages using the `requirements_py3.6.txt` or `requirements_py3.8.txt` file, depending on your Python version:

  ```bash
  pip install -r requirements_py3.[6|8].txt
  ```

#### 2. Data Preparation

- Navigate to the `data/` directory and ensure that your dataset files are correctly placed. The project expects specific data files to be present in this directory for processing.

### 3. Running the Python Library (`churn_library.py`)

- This library encapsulates the core functionality including data preprocessing, model training, and predictions.
- Execute the library script from the command line:

  ```bash
  ipython churn_library.py
  ```

- Upon running, the script will preprocess the data, train the model(s), evaluate performance, and save the output models to the models/ directory.

### 5. Testing and Logging (`churn_script_logging_and_tests.py`)

- To ensure the reliability and stability of the project code, run the logging and test script:

  ```bash
  python churn_script_logging_and_tests.py
  ```

- This script will execute a series of tests on the functions within the `churn_library.py` and log the outcomes, highlighting any errors or issues in the process.
