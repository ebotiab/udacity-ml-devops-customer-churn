import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


class DataAnalyzer:
    def __init__(self, df, filepath):
        """
        Class to load and analyze data from a specified file.
        It will generate visualizations and store the results in the images folder.
        """
        self.filepath = filepath
        self.data = self.import_data()

    def import_data(self):
        '''
        returns dataframe for the csv found at pth

        input:
                None
        output:
                df: pandas dataframe
        '''	
        pass

    def perform_eda(self):
        '''
        perform eda on self.data and save figures to images folder
        input:
                None

        output:
                None
        '''
        pass

    def encoder_helper(self, category_lst: list[str], response: str | None = None):
        '''
        helper function to turn each categorical column into a new column with
        proportion of churn for each category - associated with cell 15 from the notebook

        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                df: pandas dataframe with new columns for
        '''
        pass