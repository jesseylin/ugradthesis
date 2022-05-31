#!/usr/bin/env python

# required libraries
import os
import sys
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpinSequenceDataset(Dataset):
    """ Class for Ising spin sequence datasets """

    def __init__(self, data_filename: str, interaction_K: list):
        self.x, self.y = self.setup(data_filename)
        self.tensor_dataset = TensorDataset(self.x, self.y)
        self.interaction_K = interaction_K

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, index):
        return self.tensor_dataset[index]

    def setup(self, data_filename: str):
        file_extension = os.path.splitext(data_filename)[-1]

        if file_extension == ".txt":
            # do 2D from 1 lattice
            lattice_array = load_isinggpu_lattice(data_filename)
            lattice_tensor = torch.from_numpy(lattice_array).long().unsqueeze(-1)

            x = lattice_tensor[:, :-1].float()
            y = torch.Tensor.long((lattice_tensor[:, 1:] + 1) / 2)
            return x, y

        elif file_extension == ".npz":
            # 1D datasets
            loaded_data = np.load(data_filename)
            if len(loaded_data.items()) > 1:
                raise RuntimeError("Datafile has extra dict entries. Check manually.")
            else:
                sequences_array, = list(loaded_data.values())
                sequences_tensor = torch.from_numpy(sequences_array).long().unsqueeze(-1)

            x = sequences_tensor[:, :-1].float()
            y = torch.Tensor.long((sequences_tensor[:, 1:] + 1) / 2)

            # returns array with dimensions number of samples x length of sample x number of features (i.e., 1)
            return x, y

        elif file_extension == ".pt":
            # do 2D from the saved pytorch tensors
            lattice_tensor = torch.load(data_filename)
            x = lattice_tensor[:, :-1].float()
            y = torch.Tensor.long((lattice_tensor[:, 1:] + 1) / 2)

            return x, y


def make_output_folder(foldername_str: str, data_dir: str = "."):
    now = datetime.now()
    time_string = now.strftime("%Y%m%d")
    # TODO: Add back incrementing to prevent duplicates, using the while loop
    #    increment_string = "0"
    #    folder_name = f"{time_string}_{foldername_str}_{increment_string}"
    folder_name = f"{time_string}_{foldername_str}"
    folder_filepath = os.path.join(data_dir, folder_name)
    if os.path.exists(folder_filepath):
        raise Exception("Folder exists, exiting.")
    else:
        os.mkdir(folder_filepath)
    return folder_filepath


def save_npz(filename_str, arr, data_dir=None):
    """ Saves NPZ file """
    now = datetime.now()
    time_string = now.strftime("%Y%m%d")
    increment_string = "0"
    data_filename = f"{time_string}_{filename_str}_{increment_string}"
    data_filepath = os.path.join(data_dir, data_filename + ".npz")
    while os.path.exists(data_filepath):
        increment_string = str(int(increment_string) + 1)
        data_filename = f"{filename_str}_{increment_string}"
        data_filepath = os.path.join(data_dir, time_string + "_" + data_filename + ".npz")
    np.savez(data_filepath, data=arr)
    print("Saved file:", data_filepath)
    return


def load_isinggpu_lattice(_filename, num_rows=None):
    """ Loads lattices stored by ising-gpu into numpy ndarray """
    data = np.empty((2048, 2048))
    f = open(_filename)
    for i, l in enumerate(f):
        if num_rows is not None:
            if i == num_rows:
                break
        data[i] = [int(c) for c in l.strip(" \n\r")]
    data[data == 0] = -1
    return data


# PCA implementation
def covariance(X):
    """
    Computes the covariance, assuming the rows are examples of the data.
    :param X:
    :return:
    """
    return np.dot(X.T, X) / X.shape[0]


def pca(data, pc_count=2):
    standardized_data = np.copy(data)
    standardized_data -= np.mean(data, axis=0)
    standardized_data /= np.std(data, axis=0)
    cov_mat = covariance(standardized_data)
    evals, evecs = np.linalg.eigh(cov_mat)
    sort_indices = np.argsort(evals)[::-1][:pc_count]
    reduced_evals, reduced_evecs = evals[sort_indices], evecs[:, sort_indices]

    low_dim_projection = np.dot(data, reduced_evecs)
    return low_dim_projection, evals, evecs


# file wrangling helpers
def get_K(filename):
    """ Parse out the value of interaction_K from the rnn compiled tight filenames """
    return os.path.basename(filename).split("_")[3].split("=")[-1]


def get_T(filename):
    """ Parse out the value of T from the compiled tensors filenames and the ising-gpu lattice txts """
    filename, file_ext = os.path.splitext(filename)
    if file_ext == ".pt":
        tmp = os.path.basename(filename).split("_")[5]
        a = tmp.index("=")
        temperature = tmp[a + 1:]
    elif file_ext == ".txt":
        tmp = os.path.basename(filename).split("_")
        t_index = tmp.index("T")
        temperature = tmp[t_index + 1]
    return temperature


class IsingRNN_compat(pl.LightningModule):
    def __init__(self,
                 hidden_size, num_layers, nonlinearity="tanh", bias_on=True,
                 learning_rate=1e-2, loss_fn=nn.NLLLoss(),
                 output_data_dir="."):
        super().__init__()

        # Defining some parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.bias_on = bias_on
        self.output_data_dir = output_data_dir
        self.save_hyperparameters()

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, nonlinearity=nonlinearity, bias=bias_on)

        # Fully connected layer
        self.fc = nn.Linear(in_features=hidden_size, out_features=2, bias=bias_on)

        # Return logits
        self.softmax = nn.LogSoftmax(dim=2)

        # Define loss history
        self.loss_history = []

    def forward(self, x):
        out, hidden = self.rnn(x)
        hidden_sequence = out

        out = self.fc(out)
        out = self.softmax(out)

        return out, hidden, hidden_sequence

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _, _ = self(x)

        loss = 0
        # compute loss over time
        for i in range(y.shape[1]):
            loss += self.loss_fn(y_hat[:, i, :], y[:, i, :].view(-1)) / y.shape[1]
        self.log("train_loss", loss)
        return loss

    def predict(self, spin, hidden=None):
        """ Predict the next spin probabilistically """
        with torch.no_grad():
            self.eval()
            if hidden is not None:
                out, hidden = self.rnn(spin, hidden)
            else:
                out, hidden = self.rnn(spin)

            out = self.fc(out)
            out = self.softmax(out)

            probs = np.squeeze(np.exp(out.numpy()))

            next_spin = np.random.choice([-1, 1], size=(1, 1, 1), p=probs)
            next_spin = torch.from_numpy(next_spin).float()

        return next_spin, hidden

    def generate_sequence(self, sequence_length: int):
        init_spin = np.random.choice([-1, 1], size=(1, 1, 1))
        init_spin = torch.from_numpy(init_spin).float()

        spin_sequence = [init_spin]
        hidden_sequence = []

        for i in range(sequence_length - 1):
            if i == 0:
                assert init_spin == spin_sequence[-1]
                next_spin, hidden = self.predict(init_spin)
            else:
                next_spin, hidden = self.predict(spin_sequence[-1], hidden_sequence[-1])

            spin_sequence.append(next_spin)
            hidden_sequence.append(hidden)

        return torch.concat(spin_sequence, dim=0).squeeze(), torch.concat(hidden_sequence, dim=0).squeeze()

    def record_hidden(self, input_sequence):
        """ Records h(t) given a particular input sequence """
        with torch.no_grad():
            self.eval()
            _, _, hidden_sequence = self(input_sequence.view(1, -1, 1))
        return hidden_sequence


class IsingRNN(pl.LightningModule):
    def __init__(self,
                 hidden_size, num_layers, nonlinearity="tanh", bias_on=True,
                 learning_rate=1e-2, loss_fn=nn.NLLLoss(),
                 output_data_dir="."):
        super().__init__()

        # Defining some parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.bias_on = bias_on
        self.output_data_dir = output_data_dir
        self.save_hyperparameters()

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, nonlinearity=nonlinearity, bias=bias_on)

        # Fully connected layer
        self.fc = nn.Linear(in_features=hidden_size, out_features=2, bias=bias_on)

        # Return logits
        self.softmax = nn.LogSoftmax(dim=2)

        # Define loss history
        # need to register buffer later so it saves in state_dict
        self.register_buffer("loss_history", torch.tensor([], requires_grad=False), persistent=True)

    def forward(self, x):
        out, hidden = self.rnn(x)
        hidden_sequence = out

        out = self.fc(out)
        out = self.softmax(out)

        return out, hidden, hidden_sequence

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _, _ = self(x)

        loss = 0
        # compute loss over time
        for i in range(y.shape[1]):
            loss += self.loss_fn(y_hat[:, i, :], y[:, i, :].view(-1)) / y.shape[1]
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        """ Logs all losses in a stupid way """
        loss_list = torch.Tensor([list(x.values())[0].item() for x in training_step_outputs]).to(device)
        epoch_num = self.current_epoch
        save_filepath = os.path.join(self.output_data_dir, f"losses_{epoch_num}")

        # save to model file and to separate file
        self.loss_history = torch.cat((self.loss_history, loss_list))
        return

    def load_state_dict(self, state_dict, strict=True):
        self.loss_history.resize_(state_dict["loss_history"].shape)
        super().load_state_dict(state_dict, strict)
        return

    def predict(self, spin, hidden=None):
        """ Predict the next spin probabilistically """
        with torch.no_grad():
            self.eval()
            if hidden is not None:
                out, hidden = self.rnn(spin, hidden)
            else:
                out, hidden = self.rnn(spin)

            out = self.fc(out)
            out = self.logprob(out)

            probs = np.squeeze(np.exp(out.numpy()))

            next_spin = np.random.choice([-1, 1], size=(1, 1, 1), p=probs)
            next_spin = torch.from_numpy(next_spin).float()

        return next_spin, hidden

    def generate_sequence(self, sequence_length: int):
        init_spin = np.random.choice([-1, 1], size=(1, 1, 1))
        init_spin = torch.from_numpy(init_spin).float()

        spin_sequence = [init_spin]
        hidden_sequence = []

        for i in range(sequence_length - 1):
            if i == 0:
                assert init_spin == spin_sequence[-1]
                next_spin, hidden = self.predict(init_spin)
            else:
                next_spin, hidden = self.predict(spin_sequence[-1], hidden_sequence[-1])

            spin_sequence.append(next_spin)
            hidden_sequence.append(hidden)

        return torch.concat(spin_sequence, dim=0).squeeze(), torch.concat(hidden_sequence, dim=0).squeeze()

    def record_hidden(self, input_sequence):
        """ Records h(t) given a particular input sequence """
        with torch.no_grad():
            self.eval()
            _, _, hidden_sequence = self(input_sequence.view(1, -1, 1))
        return hidden_sequence


class LogisticLayer(nn.Module):
    def forward(self, x):
        if x.shape[-1] > 1:
            raise ValueError(
                "LogisticLayer only accepts scalar outputs (i.e., need to unsqueeze last dimension).")
        boltzmann = torch.exp(x)
        denominator = torch.add(1, boltzmann)
        prob = torch.divide(torch.tensor([1], dtype=float).to(device), denominator)

        # convert into two log classification probabilities
        return torch.log(torch.concat([prob, 1 - prob], dim=-1))


class IsingRNN_simple_compat(IsingRNN_compat):
    def __init__(self, *args, **kwargs):
        """ Initializes IsingRNN, but overrides fc layer and adds logprob layer """
        super().__init__(*args, **kwargs)
        del self.softmax

        # override the following network layers
        # Fully connected layer
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1, bias=self.bias_on)

        # Return log of probabilities
        self.logprob = LogisticLayer()

    def forward(self, x):
        out, hidden = self.rnn(x)
        hidden_sequence = out

        out = self.fc(out)
        out = self.logprob(out)

        return out, hidden, hidden_sequence


class IsingRNN_simple(IsingRNN):
    def __init__(self, *args, **kwargs):
        """ Initializes IsingRNN, but overrides fc layer and adds logprob layer """
        super().__init__(*args, **kwargs)
        del self.softmax

        # override the following network layers
        # Fully connected layer
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1, bias=self.bias_on)

        # Return log of probabilities
        self.logprob = LogisticLayer()

    def forward(self, x):
        out, hidden = self.rnn(x)
        hidden_sequence = out

        out = self.fc(out)
        out = self.logprob(out)

        return out, hidden, hidden_sequence


if __name__ == "__main__":
    print("Do not run this file like this.")
