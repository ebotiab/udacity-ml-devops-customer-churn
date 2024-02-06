from typing import Any

import shap
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve, classification_report

class ModelTrainer:

    def __init__(self, data: pd.DataFrame, model: Any):
        '''
        Class to perform feature engineering, train and evaluate models
        It will generate visualizations and store the results in the images folder.
        '''
        self.data = data
        self.model = model
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.feature_importance = None


    def perform_feature_engineering(self, response: str | None = None):
        '''
        input:
            response: string of response name [optional argument that could be used for naming variables or index y column]
        output:
            None
        '''

    def classification_report_image(self, out_filepath: str):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
            out_filepath: path to store the figure
        output:
            None
        '''
        pass


    def feature_importance_plot(self, out_filepath: str):
        '''
        creates and stores the feature importance in pth
        input:
            out_filepath: path to store the figure
        output:
            None
        '''
        pass

    def train_models(self, out_filepath: str = 'rf'):
        '''
        train, store model results: images + scores, and store models
        input:
            out_filepath: path to store the trained model
        output:
            None
        '''
        pass