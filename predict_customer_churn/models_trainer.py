"""
This module contains the ModelsTrainer class which is used to train models and store results.
"""

import logging
import os
from typing import Any
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

from .config import TEST_SIZE, RANDOM_STATE

sns.set_theme()
logger = logging.getLogger(__name__)


class ModelsTrainer:
    """
    To perform feature engineering, train models and store results.
    It will generate results visualizations and store them in the desired folder.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        models_dict: dict[str, Any],
        response: str | None = None,
        keep_cols: list[str] | None = None,
    ):
        """
        input:
            df: pandas dataframe
            models_lst: list of models to train
            response: string of response name
            keep_cols: list of columns to keep after feature engineering
        output:
            None
        """
        self._feat_names = []
        self.X_train, self.X_test, self.y_train, self.y_test = (
            self.perform_feature_engineering(df, response, keep_cols)
        )
        self.models_dict = models_dict

    def encoder_helper(
        self, df: pd.DataFrame, category_lst: list[str], response: str | None = None
    ):
        """
        helper function to turn each categorical column into a new column with
        proportion of the response variable for each category
        input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
        output:
            df_encoded: pandas dataframe with encoded categorical features
        """
        response = response or df.columns[-1]

        for category in category_lst:
            category_groups = df.groupby(category)[response].mean()
            encoded_lst = [category_groups.loc[val] for val in df[category]]
            df[f"{category}_{response}"] = encoded_lst

        return df.drop(category_lst, axis=1)

    def perform_feature_engineering(
        self, df: pd.DataFrame, response: str | None = None, keep_cols=None
    ):
        """
        performs feature engineering and returns training and testing data
        input:
            response: string of response name
        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        """
        response = response or df.columns[-1]

        category_col_names = df.select_dtypes(include=["object"]).columns.to_list()
        df_encoded = self.encoder_helper(df, category_col_names, response)[keep_cols]
        self._feat_names = keep_cols or df_encoded.columns
        X = df_encoded[self._feat_names].values
        y = df[response]
        return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def classification_report_image(
        self,
        y_train_preds: np.ndarray,
        y_test_preds: np.ndarray,
        out_filepath: str,
        model_name: str = "Model",
    ):
        """
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
            out_filepath: path to store the figure
        output:
            None
        """
        train_report = str(classification_report(self.y_train, y_train_preds))
        test_report = str(classification_report(self.y_test, y_test_preds))

        plt.figure(figsize=(20, 5))
        plt.rc("figure", figsize=(5, 5))
        plt_kw = {"fontsize": 10, "fontproperties": "monospace"}
        plt.text(0.01, 1.25, f"{model_name} Train", **plt_kw)
        plt.text(0.01, 0.05, train_report, **plt_kw)
        plt.text(0.01, 0.6, f"{model_name} Test", **plt_kw)
        plt.text(0.01, 0.7, test_report, **plt_kw)
        plt.axis("off")
        plt.savefig(out_filepath, bbox_inches="tight")
        logger.info("classification report generated and saved in %s", out_filepath)

    def feature_importance_plot(self, model_name: str, out_filepath: str):
        """
        creates and stores the feature importance in pth
        input:
            out_filepath: path to store the figure
        output:
            None
        """
        # Calculate feature importances
        model = self.models_dict[model_name]
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [self._feat_names[i] for i in indices]

        # Create and save plot
        plt.figure(figsize=(20, 5))
        plt.title("Feature Importance")
        plt.ylabel("Importance")
        plt.bar(range(self.X_train.shape[1]), importances[indices])
        # Add feature names as x-axis labels
        plt.xticks(range(self.X_train.shape[1]), names, rotation=90)

        plt.savefig(out_filepath, bbox_inches="tight")
        logger.info("Feature importance plot generated and saved in %s", out_filepath)

    def plot_roc_curve(
        self, y_preds: np.ndarray, out_filepath: str, fig: Figure | None = None, model_name: str = "Model"
    ):
        """
        creates and plot the roc curve
        input:
            y_test: numpy array of true values
            y_preds: numpy array of predicted values
        output:
            None
        """
        false_positive_rate, true_positive_rate, _ = roc_curve(self.y_test, y_preds)

        if fig is None:  # create a new figure
            fig, ax = plt.subplots()
            ax = ax or plt.gca()
            ax.set_xlim((0.0, 1.0))
            ax.set_ylim((0.0, 1.05))
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic")
            # Plot the diagonal line
            ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

        fig.gca().plot(false_positive_rate, true_positive_rate, lw=2, label=model_name)
        fig.gca().legend(loc="lower right")
        fig.savefig(out_filepath, bbox_inches="tight")
        logger.info("ROC curve generated and saved in %s", out_filepath)

        return fig

    def train_models(self, out_folder_path: str):
        """
        train and store models
        input:
            out_filepath: path to store the trained model
        output:
            None
        """
        for model_name, model in self.models_dict.items():
            model.fit(self.X_train, self.y_train)
            model = model.best_estimator_ if isinstance(model, GridSearchCV) else model

            model_filepath = os.path.join(out_folder_path, f"{model_name}_model.pkl")
            joblib.dump(model, model_filepath)
            logger.info("model trained saved in %s", model_filepath)

    def evaluate_models(self, out_folder_path: str):
        """
        computes the results of the models and stores them in the desired folder
        input:
            out_folder_path: path to store the results
        output:
            None
        """
        roc_fig = None
        roc_img_filepath = os.path.join(out_folder_path, "roc_curve.png")

        for model_name, model in self.models_dict.items():
            file_stem = os.path.join(out_folder_path, model_name)

            y_train_preds = model.predict(self.X_train)  # type: ignore
            y_test_preds = model.predict(self.X_test)  # type: ignore

            report_filepath = f"{file_stem}_cl_report.png"
            self.classification_report_image(
                y_train_preds, y_test_preds, report_filepath, model_name
            )

            if isinstance(model, RandomForestClassifier):
                feat_imp_filepath = f"{file_stem}_feat_imp.png"
                self.feature_importance_plot(model_name, feat_imp_filepath)

            roc_fig = self.plot_roc_curve(y_test_preds, roc_img_filepath, roc_fig, model_name)
