from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pytest

from ..config import (
    RAW_DATA_PATH,
    KEEP_COLS,
    RESPONSE,
    RANDOM_STATE,
    RFC_PARAM_GRID,
    LRC_PARAMS,
    NUM_FOLDS,
)
from ..models_trainer import ModelsTrainer


@pytest.fixture
def churn_data():
    """
    load churn data
    """
    return pd.read_csv(RAW_DATA_PATH)


@pytest.fixture
def models_trainer(churn_data: pd.DataFrame):
    """
    load model_trainer
    """
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=RFC_PARAM_GRID, cv=NUM_FOLDS)
    lrc = LogisticRegression(**LRC_PARAMS)
    return ModelsTrainer(churn_data, [cv_rfc, lrc], RESPONSE, KEEP_COLS)


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
