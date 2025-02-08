import numpy as np
import pandas as pd



#Define function for loading benchmark data sets
def load_dataset(*files: str) -> np.array:
    """
    Load benchmark data set from csv file
    """
    return_list = []
    for file in files:
        path = f'{file}' if file.endswith('csv') else f'{file}.csv'
        if path.endswith('stock_data.csv'):
            #Flip data for chronological order
            data = np.asarray(pd.read_csv(path))[::-1]
        elif path.endswith('energy_data.csv'):
            data = np.asarray(pd.read_csv(path))
        else:
            #Sine signal
            temp = []
            for k in range(5):
                #Randomly drawn frequency and phase
                freq = np.random.uniform(0, 0.1)
                phase = np.random.uniform(0, 0.1)

                temp.append([np.sin(freq*j + phase) for j in range(10000)])               

            data = np.stack(np.transpose(np.asarray(temp)))

        return_list.extend([data])
    
    return return_list if len(return_list) > 1 else return_list.pop()

#Define function for basic loading data from file
def loading(*files: str) -> pd.DataFrame:
    """
    Load data from csv file
    """
    return_list = []
    for file in files:
        path = f'{file}' if file.endswith('csv') else f'{file}.csv'
        return_list.extend([pd.read_csv(path)])

    return return_list if len(return_list) > 1 else return_list.pop()
