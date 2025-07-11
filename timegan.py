import time
from itertools import chain
import numpy as np
import torch
from torch import nn, tensor
from torch.utils.data import DataLoader, TensorDataset

# Define TimeGAN's recurrent neural networks
class Embedder(nn.Module):
    def __init__(self, module_name: str, input_features: int, hidden_dim: int, num_layers: int):
        super().__init__()

        if module_name == 'gru':
            self.rnn = nn.GRU(input_size=input_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif module_name == 'lstm':
            self.rnn = nn.LSTM(input_size=input_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"Invalid module name {module_name}. Choose 'gru' or 'lstm'.")
        self.model = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                   nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.rnn(x)
        return self.model(seq)

class Recovery(nn.Module):
    def __init__(self, module_name: str, input_features: int, hidden_dim: int, num_layers: int):
        super().__init__()

        if module_name == 'gru':
            self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif module_name == 'lstm':
            self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"Invalid module name {module_name}. Choose 'gru' or 'lstm'.")
        self.model = nn.Sequential(nn.Linear(hidden_dim, input_features),
                                   nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.rnn(x)
        return self.model(seq)

class Generator(nn.Module):
    def __init__(self, module_name: str, input_features: int, hidden_dim: int, num_layers: int):
        super().__init__()

        if module_name == 'gru':
            self.rnn = nn.GRU(input_size=input_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif module_name == 'lstm':
            self.rnn = nn.LSTM(input_size=input_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"Invalid module name {module_name}. Choose 'gru' or 'lstm'.")
        self.model = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                   nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.rnn(x)
        return self.model(seq)

class Supervisor(nn.Module):
    def __init__(self, module_name: str, hidden_dim: int, num_layers: int):
        super().__init__()

        if module_name == 'gru':
            self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers-1, batch_first=True)
        elif module_name == 'lstm':
            self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers-1, batch_first=True)
        else:
            raise ValueError(f"Invalid module name {module_name}. Choose 'gru' or 'lstm'.")
        self.model = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                   nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.rnn(x)
        return self.model(seq)

class Discriminator(nn.Module):
    def __init__(self, module_name: str, hidden_dim: int, num_layers: int):
        super().__init__()

        if module_name == 'gru':
            self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=False, batch_first=True)
        elif module_name == 'lstm':
            self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=False, batch_first=True)
        else:
            raise ValueError(f"Invalid module name {module_name}. Choose 'gru' or 'lstm'.")
        # If bidirectional = True
        # self.model = nn.Linear(2*hidden_dim, 1)
        # If bidirectional = False
        self.model = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.rnn(x)
        return self.model(seq)

# Define loss functions
def embedder_loss(x: torch.Tensor, x_tilde: torch.Tensor) -> torch.Tensor:
    return 10*torch.sqrt(nn.MSELoss()(x_tilde, x))

def supervised_loss(h: torch.Tensor, h_hat_supervise: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(h_hat_supervise[:,:-1,:], h[:,1:,:])

def generator_loss(y_fake: torch.Tensor,
                   y_fake_e: torch.Tensor,
                   h: torch.Tensor,
                   h_hat_supervise: torch.Tensor,
                   x: torch.Tensor,
                   x_hat: torch.Tensor) -> torch.Tensor:
    gamma = 1
    fake = torch.ones_like(y_fake, dtype=torch.float32, device=y_fake.device, requires_grad=False)

    # 1. Unsupervised generator loss
    g_loss_u = nn.BCEWithLogitsLoss()(y_fake, fake)
    g_loss_u_e = nn.BCEWithLogitsLoss()(y_fake_e, fake)

    # 2. Supervised loss
    g_loss_s = nn.MSELoss()(h_hat_supervise[:,:-1,:], h[:,1:,:])

    # 3. Two moments
    g_loss_v1 = torch.mean(torch.abs(torch.sqrt(torch.std(x_hat, dim=0)) - torch.sqrt(torch.std(x, dim=0))))
    g_loss_v2 = torch.mean(torch.abs(torch.mean(x_hat, dim=0) - torch.mean(x, dim=0)))
    g_loss_v = g_loss_v1 + g_loss_v2

    return g_loss_u + gamma*g_loss_u_e + 100*torch.sqrt(g_loss_s) + 100*g_loss_v

def discriminator_loss(y_real: torch.Tensor, y_fake: torch.Tensor, y_fake_e: torch.Tensor) -> torch.Tensor:
    gamma = 1
    valid = torch.ones_like(y_real, dtype=torch.float32, device=y_real.device, requires_grad=False)
    fake = torch.zeros_like(y_fake, dtype=torch.float32, device=y_fake.device, requires_grad=False)

    d_loss_real = nn.BCEWithLogitsLoss()(y_real, valid)
    d_loss_fake = nn.BCEWithLogitsLoss()(y_fake, fake)
    d_loss_fake_e = nn.BCEWithLogitsLoss()(y_fake_e, fake)

    return d_loss_real + d_loss_fake + d_loss_fake_e*gamma

# Define TimeGAN model
class TimeGAN(nn.Module):
    """
    TimeGAN model for time series generation
    """
    def __init__(self,
                 module_name: str = 'gru',
                 input_features: int = 1,
                 hidden_dim: int = 8,
                 num_layers: int = 3,
                 epochs: int = 1000,
                 batch_size: int = 128,
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        """
        Args:
            module_name (str): Name of the recurrent module ('gru' or 'lstm')
            input_features (int): Number of input features
            hidden_dim (int): Number of hidden units
            num_layers (int): Number of recurrent layers
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            device (str): Device to use for training ('cpu' or 'cuda')
        """       
        super().__init__()

        # Attributes
        self.module_name = module_name
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        # Networks
        self.embedder = Embedder(module_name, input_features, hidden_dim, num_layers)
        self.recovery = Recovery(module_name, input_features, hidden_dim, num_layers)
        self.generator = Generator(module_name, input_features, hidden_dim, num_layers)
        self.supervisor = Supervisor(module_name, hidden_dim, num_layers)
        self.discriminator = Discriminator(module_name, hidden_dim, num_layers)

        # Optimizers
        self.optimizer_e = torch.optim.Adam(chain(self.embedder.parameters(), self.recovery.parameters()), lr=learning_rate)
        self.optimizer_g = torch.optim.Adam(chain(self.generator.parameters(), self.supervisor.parameters()), lr=learning_rate)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)

        # Loss functions
        self.embedder_loss = embedder_loss
        self.supervised_loss = supervised_loss
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

        # Auxiliary
        self.fitting_time = None
        self.losses = []

    def fit(self, data_train: np.ndarray) -> None:
        """
        Train TimeGAN model in three subsequent training phases
        """
        self.fitting_time = time.time()
        data_train = tensor(data_train, dtype=torch.float32, device=self.device)

        # 1. Embedding network training
        print('Start Embedding Network Training')
        for epoch in range(self.epochs):
            batches_train = DataLoader(data_train, batch_size=self.batch_size, shuffle=True)

            self.train()
            loss_e = []
            for x in batches_train:
                self.optimizer_e.zero_grad()

                h = self.embedder(x)
                x_tilde = self.recovery(h)
                e_loss = self.embedder_loss(x, x_tilde)

                e_loss.backward()
                self.optimizer_e.step()

                loss_e.append(e_loss.item())

            if (epoch + 1) % (0.1*self.epochs) == 0:
                print('\rEpoch', repr(epoch + 1).rjust(len(str(self.epochs))), 'of', self.epochs, '| loss_e', f'{np.mean(loss_e):12.9f}')

        print('Finished Embedding Network Training\n')

        # 2. Training using only supervised loss
        print('Start Training with Supervised Loss Only')
        for epoch in range(self.epochs):
            batches_train = DataLoader(data_train, batch_size=self.batch_size, shuffle=True)

            self.train()
            loss_g = []
            for x in batches_train:
                self.optimizer_g.zero_grad()

                h = self.embedder(x)
                h_hat_supervise = self.supervisor(h)

                g_loss = self.supervised_loss(h, h_hat_supervise)

                g_loss.backward()
                self.optimizer_g.step()

                loss_g.append(g_loss.item())

            if (epoch + 1) % (0.1*self.epochs) == 0:
                print('\rEpoch', repr(epoch + 1).rjust(len(str(self.epochs))), 'of', self.epochs, '| loss_g', f'{np.mean(loss_g):12.9f}')

        print('Finished Training with Supervised Loss Only\n')

        # 3. Joint training
        print('Start Joint Training')
        for epoch in range(self.epochs):
            loss_g = []
            loss_e = []
            # Traing generator twice more than discriminator
            for _ in range(2):
                dataset = TensorDataset(data_train, torch.rand(data_train.shape, dtype=torch.float32, device=self.device))

                batches_train = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

                self.train()
                for x, z in batches_train:
                    self.optimizer_g.zero_grad()

                    h = self.embedder(x)
                    e_hat = self.generator(z)
                    h_hat = self.supervisor(e_hat)
                    h_hat_supervise = self.supervisor(h)
                    x_hat = self.recovery(h_hat)
                    y_fake = self.discriminator(h_hat)
                    y_fake_e = self.discriminator(e_hat)

                    g_loss = self.generator_loss(y_fake, y_fake_e, h, h_hat_supervise, x, x_hat)

                    g_loss.backward()
                    self.optimizer_g.step()

                    loss_g.append(g_loss.item())

                    self.optimizer_e.zero_grad()

                    h = self.embedder(x)
                    h_hat_supervise = self.supervisor(h)
                    x_tilde = self.recovery(h)

                    e_loss = self.embedder_loss(x, x_tilde) + 0.1*self.supervised_loss(h, h_hat_supervise)

                    e_loss.backward()
                    self.optimizer_e.step()

                    loss_e.append(e_loss.item())

            dataset = TensorDataset(data_train, torch.rand(data_train.shape, dtype=torch.float32, device=self.device))

            batches_train = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.train()
            loss_d = []
            for x, z in batches_train:
                self.optimizer_d.zero_grad()

                h = self.embedder(x)
                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)

                y_fake = self.discriminator(h_hat)
                y_real = self.discriminator(h)
                y_fake_e = self.discriminator(e_hat)

                d_loss = self.discriminator_loss(y_real, y_fake, y_fake_e)

                loss_d.append(d_loss.item())

                if d_loss > 0.15:
                    d_loss.backward()
                    self.optimizer_d.step()

            self.losses.append([np.mean(loss_g), np.mean(loss_e), np.mean(loss_d)])

            if (epoch + 1) % (0.1*self.epochs) == 0:
                print('\rEpoch', repr(epoch + 1).rjust(len(str(self.epochs))), 'of', self.epochs,
                      '| loss_g', f'{np.mean(loss_g):12.9f}',
                      '| loss_e', f'{np.mean(loss_e):12.9f}',
                      '| loss_d', f'{np.mean(loss_d):12.9f}')

        self.fitting_time = np.round(time.time() - self.fitting_time, 3)
        print('Finished Joint Training\n')
        print('\nElapsed Training Time:', time.strftime('%Hh %Mmin %Ss', time.gmtime(self.fitting_time)), '\n')

    def transform(self, data_shape: tuple) -> np.ndarray:
        """
        Generate data using the trained TimeGAN model
        """
        batches_z = DataLoader(torch.rand(size=data_shape, dtype=torch.float32, device=self.device, requires_grad=False),
                               batch_size=1)

        generated_data = []
        self.eval()
        with torch.no_grad():
            for z in batches_z:
                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)
                x_hat = self.recovery(h_hat)

                generated_data.append(np.squeeze(x_hat.cpu().numpy(), axis=0))

        return np.stack(generated_data)
    