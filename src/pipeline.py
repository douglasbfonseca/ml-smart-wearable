"""pipeline"""
import logging

import pandas as pd
from sklearn.pipeline import Pipeline

from src.initial_transformer import DataTransformer
from src.save_results import SaveResults
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report


class MLReport():
    """
    Main Machine Learn class
    """
    def __init__(self,
                 initial_transformer: DataTransformer,
                 results: SaveResults
                 ) -> None:
        """
        Constructor for Machine Learning jobs

        :param initial_transformer: Class to get data from source and transforms it
        :param results: Class to save results
        """
        self._logger = logging.getLogger(__name__)
        self._transformer = initial_transformer
        self._results = results
        self._estimators = [{'name':'random_forest', 'obj' : RandomForestClassifier()},
                            {'name': 'xgboost', 'obj': XGBClassifier()}]
        self._pipe_pre = Pipeline([('pre_sc_std', StandardScaler())])
    
    def spliter(self, data_frame: pd.DataFrame, estimator_name: str) -> tuple[pd.DataFrame]:
        """
        Splits data

        :param data_frame: Pandas DataFrame
        :param estimator_name: Name of the Estimator

        returns:
            X_train: Features training data
            X_test: Features test data
            y_train: Targuet training data
            y_test: Targuet test data
        """

        self._logger.info('Spliting data')
        X = data_frame.drop(columns=['atividade'])
        y = data_frame['atividade']
        if estimator_name == 'xgboost':
            y = y.replace({'Downstairs': 0,
                           'Jogging': 1,
                           'Sitting': 2,
                           'Standing': 3,
                           'Upstairs': 4,
                           'Walking': 5})

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)

        return X_train, X_test, y_train, y_test

    def run_pipeline(self):
        """
        Calls funcions to run the pipelines
        """
        # Getting data
        data_frame = self._transformer.data_transformer()
        self._logger.info('Data taken')
        
        for estimator in self._estimators:
            # Spliting
            X_train, X_test, y_train, y_test = self.spliter(data_frame, estimator['name'])
            
            # Last pipe
            pipe_estimator = Pipeline([('pipe_pre', self._pipe_pre),
                                       ('estimator', estimator['obj'])])
            
            # Fitting
            pipe_estimator.fit(X_train, y_train)
            
            # Predict
            y_pred = pipe_estimator.predict(X_test)

            # Report
            print('Classification report:', estimator['name'])
            print(classification_report(y_test, y_pred))

            #Saving results
            cr = classification_report(y_test, y_pred, output_dict=True)
            df_cr = pd.DataFrame(cr).transpose()
            self._results.cr_writer(estimator['name'], df_cr)
            self._results.model_writer(estimator['name'], pipe_estimator)
        
        return True