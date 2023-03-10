"""initial_transformer"""

import pandas as pd

class DataTransformer():
    """
    Class for get data from source and transforms it
    """
    def __init__(self, path: str, overlap: float, window_param: int) -> None:
        """
        Constructor of data transformer

        :param path: path of the data
        """
        self._path = path
        self._overlap = overlap
        self._window_param = window_param

    def data_transformer(self) -> pd.DataFrame:
        """
        Gets data from source and transforms it
        """
        #Getting data
        columns = ['individuo', 'atividade', 'timestamp', 'a_x', 'a_y', 'a_z']
        data_frame = pd.read_csv(self._path, sep=',', header=None, on_bad_lines='skip')
        
        #Filling NaN values with immediately preceding value
        data_frame = data_frame.fillna(method='backfill')

        #Applying columns names
        data_frame.columns = columns

        #Converting time to seconds
        data_frame['timestamp'] = data_frame['timestamp'].apply(lambda x: x/1e9)

        #Removing ';' and coverting a_z to float64
        data_frame['a_z'] = data_frame['a_z'].apply(lambda x: float(x.replace(';', '')) if type(x) == str else x)

        #Sorting and reseting index
        data_frame = data_frame.sort_values(by=['individuo','timestamp'])
        data_frame = data_frame.reset_index(drop=True)

        #To get a 3 seconds window with 20Hz frequency, we need 60 observartions
        #Using moving average
        data_frame['ma_a_x'] = data_frame['a_x'].rolling(self._window_param).mean()
        data_frame['ma_a_y'] = data_frame['a_y'].rolling(self._window_param).mean()
        data_frame['ma_a_z'] = data_frame['a_z'].rolling(self._window_param).mean()

        #Dropping unneeded columns
        data_frame = data_frame.drop(columns=['a_x', 'a_y', 'a_z'])

        #Dropping NaN values after moving average use
        data_frame = data_frame[59:].reset_index(drop=True)

        #Overlap (0 -> no overlap / 1 -> 100% overlap)
        overlap_param = int(60 - (60 * self._overlap))
        data_frame = data_frame[(data_frame.index) % overlap_param == 0]

        return data_frame