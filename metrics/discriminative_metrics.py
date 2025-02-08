import itertools
from itertools import chain
import numpy as np
import torch
from torch import nn, tensor
from torch.utils.data import DataLoader, TensorDataset
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tqdm
from tqdm import tqdm



#Define post-hoc discriminator
class discriminator(nn.Module):
    def __init__(self, input_features, hidden_dim, epochs, batch_size, device):
        super().__init__()
        #Parameters
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        #Layers
        self.rnn = nn.GRU(input_size=input_features, hidden_size=hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.model = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()

        #Optimizer
        self.optimizer = torch.optim.Adam(chain(self.rnn.parameters(), self.model.parameters()))

        #Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()        

    def forward(self, x):
        _, d_last_states = self.rnn(x)
        y_hat_logit = self.model(torch.swapaxes(d_last_states, 0, 1))
        y_hat = self.activation(y_hat_logit)
        return y_hat_logit, y_hat

    def fit(self, x, x_hat):
        """
        Train model on real and synthetic data and test on both to evaluate classification accuracy
        """
        #Split data into train and test fractions
        x_train, x_test, x_hat_train, x_hat_test = train_test_split(x, x_hat, test_size=0.2)

        x_train, x_hat_train = tensor(x_train, dtype=torch.float32, device=self.device), tensor(x_hat_train, dtype=torch.float32, device=self.device)
        dataset_train = TensorDataset(x_train, x_hat_train)

        x_test, x_hat_test = tensor(x_test, dtype=torch.float32, device=self.device, requires_grad=False), tensor(x_hat_test, dtype=torch.float32, device=self.device, requires_grad=False)

        for itt in tqdm(range(self.epochs)):
            batches = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)

            self.train()
            for x, x_hat in batches:
                self.optimizer.zero_grad()

                y_logit_real, y_pred_real = self.forward(x)
                y_logit_fake, y_pred_fake = self.forward(x_hat)

                d_loss_real = torch.mean(self.loss_fn(y_logit_real,
                                                      torch.ones_like(y_logit_real, dtype=torch.float32, device=self.device, requires_grad=False)))
                d_loss_fake = torch.mean(self.loss_fn(y_logit_fake,
                                                      torch.zeros_like(y_logit_fake, dtype=torch.float32, device=self.device, requires_grad=False)))

                d_loss = d_loss_real + d_loss_fake

                d_loss.backward()
                self.optimizer.step()

        self.eval()
        with torch.no_grad():
            _, y_pred_real = self.forward(x_test)
            _, y_pred_fake = self.forward(x_hat_test)

            y_pred_final = np.squeeze(np.concatenate((y_pred_real.cpu().detach().numpy(), y_pred_fake.cpu().detach().numpy()), axis=0))
            y_label_final = np.concatenate((np.ones([len(y_pred_real,)]), np.zeros([len(y_pred_fake,)])), axis=0)

            acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
            discriminative_score = np.abs(0.5 - acc)

        return discriminative_score

#Define function for computing discriminative score
def discriminative_score_metrics(ori_data: np.ndarray, generated_data: np.ndarray, device: str):
    """
    Use post-hoc RNN to classify original data and synthetic data

    Args:
        - ori_data: original data
        - generated_data: generated synthetic data

    Returns:
        - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    no, seq_len, dim = ori_data.shape

    hidden_dim = int(dim/2)
    iterations = 2000
    batch_size = 128

    model = discriminator(input_features=dim, hidden_dim=hidden_dim, epochs=iterations, batch_size=batch_size, device=device).to(device)

    discriminative_score = model.fit(ori_data, generated_data)

    return discriminative_score
