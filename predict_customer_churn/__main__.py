# library doc string
"""
This is the churn library and it provides a function to analyze and predict customer churn.
"""

import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from .config import (
    RAW_DATA_PATH,
    VALUES_MAP_COLS,
    RENAME_COLS,
    IMAGES_EDA_PATH,
    IMAGES_RESULTS_PATH,
    COLS_TO_PLOT,
    PLT_FIGSIZE,
    RESPONSE,
    KEEP_FEATS,
    RANDOM_STATE,
    RFC_PARAM_GRID,
    LRC_PARAMS,
    NUM_FOLDS,
)
from .data_analyzer import DataAnalyzer
from .models_trainer import ModelsTrainer

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def main():
    """
    main function to perform EDA, train models and store results
    """

    # perform EDA
    data_analyzer = DataAnalyzer(RAW_DATA_PATH, VALUES_MAP_COLS, RENAME_COLS)
    data_analyzer.perform_eda(
        out_folder=IMAGES_EDA_PATH, cols_to_plot=COLS_TO_PLOT, figsize=PLT_FIGSIZE
    )

    # load models
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=RFC_PARAM_GRID, cv=NUM_FOLDS)
    lrc = LogisticRegression(**LRC_PARAMS)

    # train models and save results
    churn_data = data_analyzer.data.copy()
    models_trainer = ModelsTrainer(churn_data, [cv_rfc, lrc], RESPONSE, KEEP_FEATS)
    models_trainer.train_models(IMAGES_RESULTS_PATH)


if __name__ == "__main__":
    main()
