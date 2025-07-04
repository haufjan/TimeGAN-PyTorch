import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

def preprocessing(data, sequence_length: int, use_scaling: bool = True, use_shuffling: bool = True, random_state: int = None) -> np.ndarray | tuple:
    """
    Conduct preprocessing: scale data, slice data into sequences and shuffle data stack
    
    Args:
        data: Raw input data
        sequence_length (int): Length of the sequences to be created
        use_scaling (bool): Default = True; Apply Min-Max scaling
        use_shuffling (bool): Default = True; Shuffle data stack
        random_state (int): Default = None; No random state
    Returns:
        np.ndarray | tuple: Preprocessed data stack or tuple of data stack, max values, and min values
    """
    # Minimum-Maximum scaling
    if use_scaling:
        scaler = MinMaxScaler().fit(data)
        data = scaler.transform(data)
        print('\nMaximum values:\n', scaler.data_max_, '\nMinimum values:\n', scaler.data_min_)

    # Create list of sequences defined by sequence_length and stack to a 3-dimensional array (batch, sequence_length, feature)
    data_stack = np.stack([data[i:i+sequence_length] for i in range(len(data) - sequence_length)])

    if use_shuffling:
        # In TimeGAN code the data set is mixed to make it similar to independent and identically distributed (iid)
        data_stack = shuffle(data_stack, random_state=random_state)

    return (data_stack, scaler.data_max_, scaler.data_min_) if use_scaling else data_stack