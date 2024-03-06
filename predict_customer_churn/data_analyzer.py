"""
This module contains the DataAnalyzer class which is used to load and analyze data.
"""

import logging
from typing import Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Load and analyze data from a specified file.
    It will generate visualizations and store the results in the desired folder.
    """

    def __init__(
        self,
        filepath: str,
        values_map: dict[str, dict[Any, Any]] | None = None,
        rename_cols: dict[str, str] | None = None,
    ):
        """
        input:
            filepath: string of the file path
        output:
            None
        """
        df = pd.read_csv(filepath)
        if values_map:
            df = df.replace(values_map)
        if rename_cols:
            df = df.rename(columns=rename_cols)
        self.data = df

    def perform_eda(
        self,
        out_folder: str,
        cols_to_plot: list[str] | None = None,
        figsize: Tuple[float, float] = (20, 10),
    ):
        """
        perform Exploratory Data Analysis on self.data and save figures to out_folder
        input:
            out_folder: string of the folder name to save the figures
            cols_to_plot: list of columns to plot their distribution
            figsize: tuple of figure size
        output:
            None
        """
        self.plot_data_basic_info(out_folder, figsize)

        for col in cols_to_plot or []:
            self.plot_distribution(col, out_folder)

        self.plot_correlation(out_folder)

    def plot_data_basic_info(
        self, out_folder: str, figsize: Tuple[float, float] = (20, 10)
    ):
        """
        plot basic information about the data, plot it and save the figure
        input:
            out_folder: string of the folder name to save the figures
            figsize: tuple of figure size
        output:
            None
        """
        data_head = self.data.head().to_string()
        data_shape = str(self.data.shape)
        missing_values = self.data.isnull().sum().to_string()
        data_description = self.data.describe().to_string()
        basic_info_txt = f"""Performing EDA
        \n- Display the first 5 rows of the data:\n{data_head}
        \n- Display the data dimensions:\n{data_shape}
        \n- Display the number of missing values for each column:\n{missing_values}
        \n- Display description of the data:\n{data_description}
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.text(0.5, 1, basic_info_txt, transform=ax.transAxes)
        plt.savefig(f"{out_folder}/data_basic_info.png", bbox_inches="tight")
        logger.info(
            "Basic data info has been saved in %s/data_basic_info.png", out_folder
        )

    def plot_distribution(
        self, col_name: str, out_folder: str, figsize: Tuple[float, float] = (20, 10)
    ):
        """
        plot histogram or bar chart of the specified column and save the figure
        input:
            col_name: string of the column name to plot
            out_folder: string of the folder name to save the figures
            figsize: tuple of figure size
        output:
            None
        """
        col_type = self.data[col_name].dtype
        plt.figure(figsize=figsize)
        if col_type in ["int64", "float64"]:
            sns.histplot(self.data[[col_name]], stat="density", kde=True)
        elif col_type == "object":
            sns.countplot(self.data[[col_name]], x=col_name)
        else:
            logger.warning("Col %s has %s type invalid to plot", col_name, col_type)
            return
        plt.title(f"Distribution of {col_name}")
        img_filepath = f"{out_folder}/{col_name}_distribution.png"
        plt.savefig(img_filepath)
        logger.info("Distribution saved in %s", img_filepath)

    def plot_correlation(
        self, out_folder: str, figsize: Tuple[float, float] = (20, 10)
    ):
        """
        plot correlation matrix and save the figure
        input:
            out_folder: string of the folder name to save the figures
            figsize: tuple of figure size
        output:
            None
        """
        plt.figure(figsize=figsize)
        numeric_cols = self.data.select_dtypes(
            include=["float", "int"]
        ).columns.to_list()
        sns.heatmap(
            self.data[numeric_cols].corr(), annot=False, cmap="Dark2_r", linewidths=2
        )
        plt.title("Correlation Matrix")
        img_filepath = f"{out_folder}/correlation_matrix.png"
        plt.savefig(img_filepath)
        logger.info("Correlation matrix saved in %s", img_filepath)
