"""save results"""

import pandas as pd
import pickle


class SaveResults():
    """
    Class to save models and their results
    """
    def __init__(self) -> None:
        """
        Constructor of save results
        """

    def cr_writer(self, name: str, df_cr: pd.DataFrame) -> None:
        """
        Saves Classification Report

        :param name: the name of the ML model
        :param df_cr: Pandas DataFrame with classification report output
        """
        if name == 'xgboost':
            df_cr = df_cr.rename(index={'0': 'Downstairs',
                                        '1': 'Jogging',
                                        '2': 'Sitting',
                                        '3': 'Standing',
                                        '4': 'Upstairs',
                                        '5': 'Walking'})
    
        name_cr = 'data/' + name + '_results.csv'
        df_cr.to_csv(name_cr)
        return True

    def model_writer(self, name: str, model: any) -> None:
        """
        Saves Model

        :param model: the name of the ML model
        :param model: a ML model trained
        """
        name_pkl = 'data/' + name + '_model.pkl'
        with open(name_pkl, 'wb') as file:
            pickle.dump(model, file)
        return True
    
