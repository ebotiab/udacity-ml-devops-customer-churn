import logging
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class DataAnalyzer:
    def __init__(self, filepath):
        """
        Class to load and analyze data from a specified file.
        It will generate visualizations and store the results in the images folder.
        """
        self.logger = logging.getLogger(__name__)
        self.filepath = filepath
        self.data = self.import_data()

    def import_data(self):
        """returns dataframe for the csv found at pth"""
        return pd.read_csv(self.filepath)

    def perform_eda(self, out_filepath: str):
        """perform eda on self.data and save figures to images folder"""

        data_head = self.data.head().to_string()
        data_shape = str(self.data.shape)
        missing_values = self.data.isnull().sum().to_string()
        data_description = self.data.describe().to_string()
        full_text = f"""Performing EDA

        - Display the first 5 rows of the data:
        {data_head}

        - Display the data dimensions:
        {data_shape}

        - Display the number of missing values for each column:
        {missing_values}

        - Display description of the data:
        {data_description}
        """
        print(full_text)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.text(0.5, 1, full_text, fontsize=10, transform=ax.transAxes)

        # Save the figure
        plt.savefig(out_filepath, bbox_inches="tight")
        plt.close(fig)

    def plot_distribution(
        self, col_name: str, out_filepath: str, figsize: Tuple[float, float] = (20, 10)
    ):
        """plot histogram or bar chart of the specified column and save the figure"""
        self.logger.debug(f"Display the distribution of {col_name} col:")
        plt.figure(figsize=figsize)
        col_type = self.data[col_name].dtype
        if col_type in ["int64", "float64"]:
            sns.histplot(self.data[col_name], stat="density", kde=True)
        elif col_type == "object":
            sns.countplot(self.data[col_name])
        else:
            self.logger.warning(
                f"Column {col_name} has {col_type} type, which is not a valid type for distribution plot"
            )
            return
        plt.title(f"Distribution of {col_name}")
        plt.savefig(f"{out_filepath}/{col_name}_distribution.png")
        plt.show()

    def plot_correlation(
        self, out_filepath: str, figsize: Tuple[float, float] = (20, 10)
    ):
        """plot correlation matrix and save the figure"""
        self.logger.debug("Display the correlation matrix")
        plt.figure(figsize=figsize)
        sns.heatmap(self.data.corr(), annot=False, cmap="Dark2_r", linewidths=2)
        plt.title("Correlation Matrix")
        plt.savefig(f"{out_filepath}/correlation_matrix.png")
        plt.show()

    def encoder_helper(self, category_lst: list[str], response: str):
        """
        helper function to turn each categorical column into a new column with
        proportion of the response variable for each category
        input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]
        output:
            None
        """
        response = response or self.data.columns[-1]

        for category in category_lst:
            category_groups = self.data.groupby(category)[response].mean()
            encoded_lst = [category_groups.loc[val] for val in self.data[category]]
            self.data[f"{category}_{response}"] = encoded_lst
