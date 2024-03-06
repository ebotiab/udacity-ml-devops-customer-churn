"""
This module contains the unit tests for the data_analyzer module.
"""

import os
import logging

import matplotlib.pyplot as plt

from .config import COLS_TO_PLOT, IMAGES_EDA_PATH, PLT_FIGSIZE, RAW_DATA_PATH
from ..predict_customer_churn import DataAnalyzer

logging.basicConfig(
    filename="./tests/logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import():
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = DataAnalyzer(RAW_DATA_PATH).data
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err
    

def test_plot_data_basic_info(data_analyzer: DataAnalyzer):
    """
    test plot data basic info function
    """
    logging.info("Plotting basic info")

    # Call the plot_data_basic_info method
    try:
        data_analyzer.plot_data_basic_info(IMAGES_EDA_PATH, PLT_FIGSIZE)
        logging.info("Basic info plot executed successfully")
    except Exception as e:
        logging.error("An error occurred during basic info plot: %s", e)
        raise e

    # Assert that the plot is saved in the output folder
    try:
        assert os.path.exists(os.path.join(IMAGES_EDA_PATH, "data_basic_info.png"))
    except AssertionError as err:
        logging.error(
            "Testing plot_data_basic_info: The expected plot was not saved in the output folder"
        )
        raise err
    
def test_plot_distribution(data_analyzer: DataAnalyzer):
    """
    test plot distribution function
    """
    logging.info("Plotting distribution")

    # Call the plot_distribution method
    try:
        for col in COLS_TO_PLOT:
            data_analyzer.plot_distribution(col, IMAGES_EDA_PATH)
        logging.info("Distribution plots executed successfully")
    except Exception as e:
        logging.error("An error occurred during distribution plots: %s", e)
        raise e

    # Assert that the plots are saved in the output folder
    try:
        for col in COLS_TO_PLOT:
            assert os.path.exists(os.path.join(IMAGES_EDA_PATH, f"{col}_distribution.png"))
    except AssertionError as err:
        logging.error(
            "Testing plot_distribution: The expected plots were not saved in the output folder"
        )
        raise err
    
def test_plot_correlation_matrix(data_analyzer: DataAnalyzer):
    """
    test plot correlation matrix function
    """
    logging.info("Plotting correlation matrix")

    # Call the plot_correlation_matrix method
    try:
        data_analyzer.plot_correlation(IMAGES_EDA_PATH)
        logging.info("Correlation matrix plot executed successfully")
    except Exception as e:
        logging.error("An error occurred during correlation matrix plot: %s", e)
        raise e

    # Assert that the plot is saved in the output folder
    try:
        assert os.path.exists(os.path.join(IMAGES_EDA_PATH, "correlation_matrix.png"))
    except AssertionError as err:
        logging.error(
            "Testing plot_correlation_matrix: The expected plot was not saved in the output folder"
        )
        raise err
