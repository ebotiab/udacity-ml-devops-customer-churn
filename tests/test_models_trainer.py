"""
This module contains tests for the models_trainer module
"""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from tests.config import IMAGES_RESULTS_PATH, RESPONSE, KEEP_FEATS
from predict_customer_churn import ModelsTrainer

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_encoder_helper(models_trainer: ModelsTrainer, churn_data: pd.DataFrame):
    """
    test encoder helper
    """
    logging.info("Performing data encoding")
    try:
        cat_columns = churn_data.select_dtypes(include=["object"]).columns.to_list()
        df_enc = models_trainer.encoder_helper(churn_data, cat_columns, RESPONSE)
        logging.info("Data encoding executed successfully")
    except Exception as e:
        logging.error("An error occurred during data encoding: %s", e)
        raise e

    # Assert that the encoded dataframe has categorical columns replaced with encoded columns
    try:
        new_cols = [
            f"{col}_{RESPONSE}" if col in cat_columns else col
            for col in churn_data.columns
        ]
        assert all(col not in df_enc.columns for col in new_cols)
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The categorical columns were not replaced with encoded columns"
        )
        raise err


def test_perform_feature_engineering(
    models_trainer: ModelsTrainer, churn_data: pd.DataFrame
):
    """
    test perform_feature_engineering
    """

    logging.info("Performing feature engineering")

    try:
        X_train, X_test, y_train, y_test = models_trainer.perform_feature_engineering(
            churn_data, RESPONSE, KEEP_FEATS
        )
        logging.info("Feature engineering executed successfully")
    except Exception as e:
        logging.error("An error occurred during feature engineering: %s", e)
        raise e
    # Assert that the encoded dataframe has KEEP_COLS columns
    try:
        assert all(col not in X_train.columns for col in KEEP_FEATS)
        assert all(col not in X_test.columns for col in KEEP_FEATS)
        assert y_train.name == RESPONSE
        assert y_test.name == RESPONSE
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The expected cols were not found in df"
        )
        raise err


def test_classification_report_image(
    models_trainer: ModelsTrainer, train_preds: np.ndarray, test_preds: np.ndarray
):
    """
    test classification_report_image
    """
    logging.info("Generating classification report")
    try:
        img_results_filepath = os.path.join("images", "results", "test_cl_report.png")
        models_trainer.classification_report_image(
            train_preds, test_preds, img_results_filepath
        )
        logging.info("Classification report generated successfully")
    except Exception as e:
        logging.error(
            "An error occurred during classification report generation: %s", e
        )
        raise e

    # Assert that the necessary plots are saved in the output folder
    try:
        assert os.path.exists(img_results_filepath)
    except AssertionError as err:
        logging.error(
            "Testing classification_report_image: The classification report image was not saved"
        )
        raise err


def test_feature_importance_plot(models_trainer: ModelsTrainer, first_model: Any):
    """
    test feature_importance_plot
    """
    logging.info("Generating feature importance plot")

    try:
        img_results_filepath = os.path.join("images", "results", "test_feat_imp.png")
        models_trainer.feature_importance_plot(first_model, img_results_filepath)
        logging.info("Feature importance plot generated successfully")
    except Exception as e:
        logging.error(
            "An error occurred during feature importance plot generation: %s", e
        )
        raise e

    # Assert that the necessary plots are saved in the output folder
    try:
        assert os.path.exists(img_results_filepath)
    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: The feature importance plot was not saved"
        )
        raise err


def test_plot_roc_curve(models_trainer: ModelsTrainer, test_preds: np.ndarray):
    """
    test plot_roc_curve
    """
    logging.info("Generating ROC curve")

    try:
        img_results_filepath = os.path.join("images", "results", "test_roc_curve.png")
        models_trainer.plot_roc_curve(test_preds, img_results_filepath)
        logging.info("ROC curve generated successfully")
    except Exception as e:
        logging.error("An error occurred during ROC curve generation: %s", e)
        raise e

    # Assert that the necessary plots are saved in the output folder
    try:
        assert os.path.exists(img_results_filepath)
    except AssertionError as err:
        logging.error("Testing plot_roc_curve: The ROC curve was not saved")
        raise err


def test_train_models(models_trainer: ModelsTrainer):
    """
    test train_models
    """
    logging.info("Training models")
    try:
        models_trainer.train_models(IMAGES_RESULTS_PATH)
        logging.info("Models trained successfully")
    except Exception as e:
        logging.error("An error occurred during model training: %s", e)
        raise e

    # Assert that the necessary plots are saved in the output folder
    try:
        assert os.path.exists(f"{IMAGES_RESULTS_PATH}/roc_curve.png")
        for model in models_trainer.models_lst:
            if isinstance(model, GridSearchCV):
                model = model.best_estimator_
            model_name = model.__class__.__name__
            file_stem = f"{IMAGES_RESULTS_PATH}/{model_name}"
            assert os.path.exists(f"{file_stem}_cl_report.png")
            if isinstance(model, RandomForestClassifier):
                assert os.path.exists(f"{file_stem}_feat_imp.png")
            assert os.path.exists(f"{file_stem}_model.pkl")
    except AssertionError as err:
        logging.error("Testing plot_roc_curve: The ROC curve was not saved")
        raise err
