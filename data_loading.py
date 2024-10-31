import pandas as pd



#Define function for loading benchmark data sets
def load_datatset(*files: str) -> pd.DataFrame:
    """Load benchmark data set from csv file."""
    return_list = []
    for file in files:
        path = f'{file}' if file.endswith('csv') else f'{file}.csv'
        if path.endswith('stock_data.csv'):
            #Flip data for chronological order
            data = pd.read_csv(path)[::-1]
        elif path.endswith('energy_data.csv'):
            data = pd.read_csv(path)
        return_list.extend([data])
        # else:
        #     #Sine
    
    return return_list if len(return_list) > 1 else return_list.pop()

#Define function for basic loading data from file
def loading(*files: str) -> pd.DataFrame:
    """Load data from csv file."""
    return_list = []
    for file in files:
        path = f'{file}' if file.endswith('csv') else f'{file}.csv'
        return_list.extend([pd.read_csv(path)])

    return return_list if len(return_list) > 1 else return_list.pop()