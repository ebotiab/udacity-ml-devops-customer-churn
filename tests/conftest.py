import os
from typing import Any
import joblib

import pandas as pd
import pytest

from .config import (
    RAW_DATA_PATH,
    VALUES_MAP_COLS,
    RENAME_COLS,
    KEEP_FEATS,
    RESPONSE,
)
from ..predict_customer_churn import DataAnalyzer, ModelsTrainer


@pytest.fixture
def data_analyzer():
    """
    load churn data
    """
    return DataAnalyzer(RAW_DATA_PATH, VALUES_MAP_COLS, RENAME_COLS)


@pytest.fixture
def churn_data(data_analyzer: DataAnalyzer):
    """
    load churn data
    """
    return data_analyzer.data


@pytest.fixture
def models_trainer(churn_data: pd.DataFrame):
    """
    load models trainer object with trained models
    """
    rfc = joblib.load(os.path.join("models", "rfc_model.pkl"))
    lrc = joblib.load(os.path.join("models", "lrc_model.pkl"))
    models = {"rfc": rfc, "lrc": lrc}
    return ModelsTrainer(churn_data, models, RESPONSE, KEEP_FEATS)


@pytest.fixture
def rfc_model(models_trainer: ModelsTrainer):
    """
    load first model
    """
    return models_trainer.models_dict["rfc"]


@pytest.fixture
def train_preds(models_trainer: ModelsTrainer, rfc_model: Any):
    """
    load train predictions
    """
    return rfc_model.predict(models_trainer.X_train)


@pytest.fixture
def test_preds(models_trainer: ModelsTrainer, rfc_model: Any):
    """
    load test predictions
    """
    return rfc_model.predict(models_trainer.X_test)
