from itertools import chain
import numpy as np
import torch
from torch import nn, tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

class Predictor(nn.Module):
    """
    Post-hoc predictor for TimeGAN
    """
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 epochs: int = 5000,
                 batch_size: int = 128,
                 device: str = 'cpu'):
        """
        Args:
            dim (int): Dimension of the data
            hidden_dim (int): Dimension of the hidden layer
            epochs (int): Number of epochs for training
            batch_size (int): Batch size for training
            device (str): Device to use for training ('cpu' or 'cuda')
        """
        super().__init__()

        # Attributes
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        # Layers
        self.rnn = nn.GRU(input_size=dim-1, hidden_size=hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.model = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())

        # Optimizer
        self.optimizer = torch.optim.Adam(chain(self.rnn.parameters(), self.model.parameters()), 1e-3)

        # Loss function
        self.loss_fn = nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_outputs, _ = self.rnn(x)
        return self.model(p_outputs)

    def fit(self, data_train: np.ndarray, data_test: np.ndarray) -> float:
        """
        Train model on synthetic and test on real data
        """
        x_train = data_train[:,:-1,:(self.dim-1)]
        y_train = np.reshape(data_train[:,1:,(self.dim-1)], (data_train.shape[0], data_train.shape[1]-1, 1))

        x_train = tensor(x_train, dtype=torch.float32, device=self.device)
        y_train = tensor(y_train, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(x_train, y_train)
        for _ in tqdm(range(self.epochs)):
            batches_train = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.train()
            # Mini batches, train on synthetic
            for X, Y in batches_train:
                self.optimizer.zero_grad()

                pred_train = self.forward(X)
                loss = self.loss_fn(pred_train, Y)

                loss.backward()
                self.optimizer.step()

        # Test model on original data      
        x_test = data_test[:,:-1,:(self.dim-1)]
        y_test = np.reshape(data_test[:,1:,(self.dim-1)], (data_test.shape[0], data_test.shape[1]-1, 1))

        x_test = tensor(x_test, dtype=torch.float32, device=self.device, requires_grad=False)
        y_test = tensor(y_test, dtype=torch.float32, device=self.device, requires_grad=False)

        mae = 0
        self.eval()
        with torch.no_grad():
            pred_test =  self.forward(x_test)

            for i in range(len(pred_test)):
                mae += mean_absolute_error(y_test[i,:,:].cpu().detach().numpy(), pred_test[i,:,:].cpu().detach().numpy())

        return mae

def predictive_score_metrics(ori_data: np.ndarray, generated_data: np.ndarray, device: str) -> float:
    """
    Report the performance of Post-hoc RNN one-step ahead prediction
    """
    # no, seq_len, dim
    no, _, dim = np.asarray(ori_data).shape

    hidden_dim = int(dim/2)
    iterations = 5000
    batch_size = 128

    # Instantiate predictor model
    model = Predictor(dim, hidden_dim, iterations, batch_size, device=device).to(device)
    # Train model and compute prdictive score
    predictive_score = model.fit(generated_data, ori_data)
    
    return predictive_score/no
