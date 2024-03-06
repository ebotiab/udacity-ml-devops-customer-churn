"""
This file contains the configuration variables used in the project.
"""

RAW_DATA_PATH = "./data/bank_data.csv"
MODELS_PATH = "./models"
IMAGES_EDA_PATH = "./images/eda"
IMAGES_RESULTS_PATH = "./images/results"

COLS_TO_PLOT = [
    'Churn',
    'Customer_Age',
    'Marital_Status',
    "Total_Trans_Ct"
]

RESPONSE = 'Churn'
PLT_FIGSIZE = (20, 5)

VALUES_MAP_COLS = {
    "Attrition_Flag": {"Existing Customer": 0, "Attrited Customer": 1},
}
RENAME_COLS = {
    "Attrition_Flag": "Churn",
}

KEEP_FEATS = [
    'Customer_Age', 'Dependent_count', 'Months_on_book', 
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
    'Income_Category_Churn', 'Card_Category_Churn'
]

TEST_SIZE = 0.3
RANDOM_STATE = 42

RFC_PARAMS = {"random_state": 42, "n_estimators": 100}
RFC_PARAM_GRID = { 
    'n_estimators': [200, 500],
    'max_features': [None, 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
}
NUM_FOLDS = 5

# Use a different solver if the default 'lbfgs' fails to converge
# Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
LRC_PARAMS = {"solver":'newton-cholesky', "max_iter": 3000}
