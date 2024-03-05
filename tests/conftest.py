from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pytest

from tests.config import (
    RAW_DATA_PATH,
    VALUES_MAP_COLS,
    RENAME_COLS,
    KEEP_FEATS,
    RESPONSE,
    RANDOM_STATE,
    RFC_PARAM_GRID,
    LRC_PARAMS,
    NUM_FOLDS,
)
from predict_customer_churn import DataAnalyzer, ModelsTrainer


@pytest.fixture
def churn_data():
    """
    load churn data
    """
    return DataAnalyzer(RAW_DATA_PATH, VALUES_MAP_COLS, RENAME_COLS).data


@pytest.fixture
def models_trainer(churn_data: pd.DataFrame):
    """
    load model_trainer
    """
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=RFC_PARAM_GRID, cv=NUM_FOLDS)
    lrc = LogisticRegression(**LRC_PARAMS)
    return ModelsTrainer(churn_data, [cv_rfc, lrc], RESPONSE, KEEP_FEATS)


@pytest.fixture
def first_model(models_trainer: ModelsTrainer):
    """
    load first model
    """
    return models_trainer.models_lst[0]


@pytest.fixture
def train_preds(models_trainer: ModelsTrainer, first_model: Any):
    """
    load train predictions
    """
    return first_model.predict(models_trainer.X_train)


@pytest.fixture
def test_preds(models_trainer: ModelsTrainer, first_model: Any):
    """
    load test predictions
    """
    return first_model.predict(models_trainer.X_test)
