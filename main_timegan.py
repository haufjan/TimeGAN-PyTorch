import argparse
import numpy as np
import torch
from data_loading import load_datatset
from timegan import TimeGAN
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import low_dimensional_representation, plot_distribution_estimate
from utils import preprocessing



def main(args):
    """Main function"""
    #Check available device
    device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
    )
    print(f'Using {device} device')

    #Load data from file
    data = load_datatset(args.data)

    #Preprocessing
    data_train, max_val, min_val = preprocessing((data, True), sequence_length=args.seq_len)

    #Instantiate TimeGAN model
    model = TimeGAN(input_features=data_train.shape[-1],
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate).to(device)

    #Start training
    model.fit(data_train)

    #Synthesize sequences
    data_gen = model.transform(data_train.shape)

    #Evaluation section
    metric_results = {}

    #Discriminative score
    discriminative_score = []
    for _ in range(args.metric_iteration):
        temp_disc = discriminative_score_metrics(data_train, data_gen)
        discriminative_score.append(temp_disc)
    
    metric_results['discriminative'] = np.mean(discriminative_score)

    #Predictive score
    predictive_score = []
    for _ in range(args.metric_iteration):
        temp_pred = predictive_score_metrics(data_train, data_gen)
        predictive_score.append(temp_pred)
    
    metric_results['predictive'] = np.mean(predictive_score)

    print(metric_results)

    #Visualization
    plot_distribution_estimate(*low_dimensional_representation(data_train, data_gen, 'pca'), 'pca')
    plot_distribution_estimate(*low_dimensional_representation(data_train, data_gen, 'tsne'), 'tsne')

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

    #Main function call
    data_train, data_gen, metrics = main(args)