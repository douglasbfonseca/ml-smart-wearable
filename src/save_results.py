"""save results"""

import pandas as pd
import pickle


class SaveResults():
    """
    Save models and their results
    """
    def __init__(self) -> None:
        """
        Constructor of save results
        """

    def cr_writer(self, name: str, df_cr: pd.DataFrame):
        """
        Saves Classification Report
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

    def model_writer(self, name: str, model: any):
        """
        Saves Model
        """
        name_pkl = 'data/' + name + '_model.pkl'
        with open(name_pkl, 'wb') as file:
            pickle.dump(model, file)
        return True
    
