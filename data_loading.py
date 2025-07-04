import numpy as np
import pandas as pd

def load_dataset(file_path: str) -> np.ndarray:
    """
    Load benchmark data set from CSV file
    """
    file_path = file_path if file_path.endswith('csv') else file_path + '.csv'
    if file_path.endswith(('stock_data.csv', 'energy_data.csv')):
        # Flip data for chronological order
        benchmark_data = np.asarray(pd.read_csv(file_path))[::-1]  
    else:
        rng = np.random.default_rng(seed=42)              
        sine_signal = []
        for _ in range(5):
            # Randomly drawn frequency and phase
            frequency = rng.uniform(0, 0.1)
            phase = rng.uniform(0, 0.1)
            sine_signal.append([np.sin(frequency * j + phase) for j in range(10000)])

        benchmark_data = np.stack(np.transpose(np.asarray(sine_signal)))
    
    return benchmark_data

def load_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame
    """
    return pd.read_csv(file_path if file_path.endswith('csv') else file_path + '.csv')
