import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_loading import load_dataset
from timegan import TimeGAN
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import low_dimensional_representation, plot_distribution_estimate
from utils import preprocessing

def get_device() -> str:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'    
    print(f'Using {device} device')
    return device

def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    # Random seeds
    np.random.seed(42)

    # Check available device
    device = get_device()

    # Load data from file
    data = load_dataset(args.data)

    # Preprocessing
    data_train, max_val, min_val = preprocessing(data, sequence_length=args.seq_len)

    # Instantiate TimeGAN model
    model = TimeGAN(input_features=data_train.shape[-1],
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate).to(device)

    # Start training
    model.fit(data_train)

    # Synthesize sequences
    data_gen = model.transform(data_train.shape)

    # Evaluation section
    metric_results = {}

    # Discriminative score
    discriminative_score = []
    for _ in range(args.metric_iteration):
        temp_disc = discriminative_score_metrics(data_train, data_gen, device)
        discriminative_score.append(temp_disc)
    
    metric_results['Discriminative'] = np.mean(discriminative_score)

    # Predictive score
    predictive_score = []
    for _ in range(args.metric_iteration):
        temp_pred = predictive_score_metrics(data_train, data_gen, device)
        predictive_score.append(temp_pred)
    
    metric_results['Predictive'] = np.mean(predictive_score)

    for metric, score in metric_results.items():
        print(f'{metric} Score: {score}')

    # Visualization
    plot_distribution_estimate(*low_dimensional_representation(data_train, data_gen, 'pca'), 'pca')
    plot_distribution_estimate(*low_dimensional_representation(data_train, data_gen, 'tsne'), 'tsne')
    plt.show()

    return data_train, data_gen, metric_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        help='Name of csv file',
        type=str
    )
    parser.add_argument(
        '--seq_len',
        help='Length of sequences',
        default=24,
        type=int
    )
    parser.add_argument(
        '--module',
        help='RNN module',
        choices=['gru', 'lstm'],
        default='gru',
        type=str
    )
    parser.add_argument(
        '--hidden_dim',
        help='Number of features for hidden vector',
        default=24,
        type=int
    )
    parser.add_argument(
        '--num_layers',
        help='Number of sequential recurrent layers',
        default=3,
        type=int
    )
    parser.add_argument(
        '--epochs',
        help='Number of iterations for training',
        default=10000,
        type=int
    )
    parser.add_argument(
        '--batch_size',
        help='Number of samples per batch during training',
        default=128,
        type=int
    )
    parser.add_argument(
        '--metric_iteration',
        help='Number of iterations for metric evaluation',
        default=10,
        type=int
    )
    parser.add_argument(
        '--learning_rate',
        help='Set learning rate for optimizer',
        default=1e-3,
        type=float
    )
    args = parser.parse_args()

    # Main function call
    real_data, generated_date, computed_metrics = main(args)
