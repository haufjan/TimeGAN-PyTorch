import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle



#Define preprocessing routine
def preprocessing(*inputs: tuple, sequence_length: int, shuffle_stack: bool = True, random_state: int = None) -> np.ndarray:
    """Conduct preprocessing: scale data, slice data into sequences and shuffle data stack.
    Consistent shuffling between multiple data stacks must be performed separetaly."""
    return_list = []
    for data, bool_scale in inputs:
        #Create Minimum-Maximum scaler
        if bool_scale:
            scaler = MinMaxScaler().fit(data)
            data = scaler.transform(data)
            print('\nMaximum values:\n', scaler.data_max_, '\nMinimum values:\n', scaler.data_min_)

        #Create list of sequences from sliding window operation defined by sequence_length and stack to a 3-dimensional array (batch, sequence_length, feature)
        data_stack = np.stack([data[i:i+sequence_length] for i in range(len(data) - sequence_length)])

        if shuffle_stack:
            #In TimeGAN code the data set is mixed to make it similar to independent and identically distributed (iid)
            data_stack = shuffle(data_stack, random_state=random_state)

        return_list.extend([data_stack, scaler.data_max_, scaler.data_min_]) if bool_scale else return_list.extend([data_stack])

    return return_list if len(return_list) > 1 else return_list.pop()