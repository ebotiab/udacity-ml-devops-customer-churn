"""
This module contains the unit tests for the data_analyzer module.
"""

import os
import logging

from .config import COLS_TO_PLOT, IMAGES_EDA_PATH, PLT_FIGSIZE, RAW_DATA_PATH
from .data_analyzer import DataAnalyzer

logging.basicConfig(
    filename="./logs/churn_library.log",
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


def test_perform_eda():
    """
    test perform eda function
    """
    logging.info("Performing EDA")

    # Create an instance of the DataAnalyzer class
    data_analyzer = DataAnalyzer(RAW_DATA_PATH)

    # Define the inputs for the perform_eda method
    out_folder = IMAGES_EDA_PATH
    cols_to_plot = COLS_TO_PLOT
    figsize = PLT_FIGSIZE

    # Call the perform_eda method
    try:
        data_analyzer.perform_eda(out_folder, cols_to_plot, figsize)
        logging.info(
            "EDA executed successfully and figures saved in %s", IMAGES_EDA_PATH
        )
    except Exception as e:
        logging.error("An error occurred during EDA: %s", e)
        raise e

    # Assert that the necessary plots are saved in the output folder
    try:
        assert os.path.exists(os.path.join(out_folder, "data_basic_info.png"))
        for col in cols_to_plot:
            assert os.path.exists(os.path.join(out_folder, f"{col}_distribution.png"))
        assert os.path.exists(os.path.join(out_folder, "correlation_matrix.png"))
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The expected plots were not saved in the output folder"
        )
        raise err
