#!/usr/bin/env python
# coding: utf-8

# In[1]:


# usual libraries
import numpy as np
import matplotlib.pyplot as plt
import functools
import time


# In[2]:


# ML libraries
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# In[3]:


# Wolff Algorithm for PBC 1D Ising Chain
def wolffAlgorithm(chainLattice, K, steps):
    size = np.size(chainLattice)
    
    for s in range(steps):
        cluster = wolffStep(chainLattice, K)        
        # flip the cluster
        for site in cluster:
            chainLattice[site] *= -1
            
def wolffStep(chainLattice, K):
    size = np.size(chainLattice)
    pAdd = 1 - np.exp(-2*K)
    # begin wolff step
    idx = np.random.randint(size)

    # initialize frontier and cluster
    frontier = {idx}
    cluster  = {idx}

    # expand the cluster
    while len(frontier) > 0:

        newFrontier = set()

        for site in frontier:

            # check if sites to left and right are to be added
            nNbr = (site + 1) % size
            if (chainLattice[site] == chainLattice[nNbr] and 
                nNbr not in cluster and 
                np.random.rand() < pAdd):

                newFrontier.add(nNbr)
                cluster.add(nNbr)
            pNbr = (site - 1) % size
            if (chainLattice[site] == chainLattice[pNbr] and 
                pNbr not in cluster and 
                np.random.rand() < pAdd):

                newFrontier.add(pNbr)
                cluster.add(pNbr)

        frontier = newFrontier.copy()
    
    return cluster

def getSamples(chainLattice, K, sampleSteps, equilTime=1500, autoCorrTime=1500):

    size = np.size(chainLattice)
    sampleArray = np.zeros((sampleSteps, size))
    for s in range(round(equilTime)):
        cluster = wolffStep(chainLattice, K)
        
        for site in cluster:
            chainLattice[site] *= -1
            
    for s in range(sampleSteps):
        
        for ss in range(round(autoCorrTime)):
            cluster = wolffStep(chainLattice, K)
            for site in cluster:
                chainLattice[site] *= -1

        sampleArray[s] = chainLattice
    
    return sampleArray

def getMeasurements(chainLattice, K, sampleSteps, observable, autoCorrTime=None):
    size = np.size(chainLattice)
    obsArray = np.zeros(sampleSteps)
    if autoCorrTime == None:
        autoCorrTime = round(1.5*size)
        equilTime = autoCorrTime
    
    for s in range(equilTime):
        cluster = wolffStep(chainLattice, K)
        
        for site in cluster:
            chainLattice[site] *= -1
            
    for s in range(sampleSteps):
        
        for ss in range(autoCorrTime):
            cluster = wolffStep(chainLattice, K)
            for site in cluster:
                chainLattice[site] *= -1

        obsArray[s] = observable(chainLattice, K)
        
    return obsArray


# In[4]:


# observables
def energy(chainLattice, K):
    size = np.size(chainLattice)
    E = 0
    
    for site in range(size):
        alignSgn = chainLattice[site]*chainLattice[(site+1)%size]
        E += -2*K*alignSgn
        
    return E

def corrAtR(chainLattice, K, r):
    size = np.size(chainLattice)
    r = r % size
    
    return chainLattice[0]*chainLattice[r]

def corrFunc(chainLattice, K):
    size = np.size(chainLattice)
    maxR = round(size/2)
    
    corrFuncArr = np.zeros(maxR)
    
    for r, _ in enumerate(corrFuncArr):
        corrFuncArr[r] = corrAtR(chainLattice, K, r)
    
    return corrFuncArr


# In[5]:


def getCorrelations(chainLattice, K, sampleSteps, autoCorrTime=None):
    size = np.size(chainLattice)
    maxR = round(size/2)
    corrArray = np.zeros((sampleSteps, maxR))
    if autoCorrTime == None:
        autoCorrTime = round(1.5*size)
        equilTime = autoCorrTime
    
    for s in range(equilTime):
        cluster = wolffStep(chainLattice, K)
        
        for site in cluster:
            chainLattice[site] *= -1
            
    for s in range(sampleSteps):
        for ss in range(autoCorrTime):
            cluster = wolffStep(chainLattice, K)

            for site in cluster:
                chainLattice[site] *= -1

        corrArray[s] = corrFunc(chainLattice, K)
        
    dataArray = np.stack(
            (np.average(corrArray, axis=0), np.std(corrArray, axis=0)/np.sqrt(sampleSteps))
        )
    xdata = np.arange(maxR)
    return (xdata, dataArray)


# In[6]:


# some wrappers
def timethis(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end-start)
        return result
    return wrapper
# statistical bootstrap wrapper


# RNN functions

class IsingRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(IsingRNN, self).__init__()

        # Defining some parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True, nonlinearity="relu")   
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        global device
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        hidden = hidden.to(device)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        out, hidden = out.to(device), hidden.to(device)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        out = self.softmax(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return hidden


# In[12]:


def loadDataset(filename, batch_size=10):
    # Load generated dataset
    data = np.load(filename)['sampleData']

    x = data[:,:-1]
    y = data[:,1:]

    # replace -1 with 0
    x = np.where(x == -1, 0, x)
    y = np.where(y == -1, 0, y)

    # reformat according to RNN requirements
    shapeX = np.shape(x)
    shapeY = np.shape(y)

    x = np.reshape(x, (shapeX[0], shapeX[1], 1))
    y = np.reshape(y, (shapeY[0], shapeY[1], 1))

    N_SAMPLES = np.size(x, axis=0)

    xData = torch.from_numpy(x).float()
    yData = torch.from_numpy(y).long()

    #yData = torch.reshape(yData, (N_SAMPLES, 1))

    train_dataset = TensorDataset(xData, yData)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader


# In[14]:


def train_loop(dataloader, model, loss_fn, optimizer):
    global device
    size = len(dataloader.dataset)
    losses = []
    # set to training mode
    model.train()
    model.to(device)
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # compute prediction and loss
        output, hidden = model(X)
        output, hidden = output.to(device), hidden.to(device)
        
        # combines sequences across batches
        output = output.view(BATCH_SIZE*X.shape[1], 2)
        loss = loss_fn(output, y.view(-1))
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        optimizer.step()
        
        if batch % 100 == 0:
            losses.append(loss.item())
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            pred, hidden = model(X)
            test_loss += loss_fn(pred, y.view(-1)).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def getTestExample(dataloader):
    dataList = list(dataloader)
    randInd = np.random.choice(np.arange(len(dataList)))
    return dataList[randInd]

# # Generative mode

# In[15]:


def randSpin():
    # flip a coin to pick a spin, compatible with RNN input requirements
    return torch.from_numpy(np.random.choice([0,1], size=(1,1,1))).float()

def predict(model, start=None):
    # predicts the next spin
    
    if start==None:
        # flip a coin to pick starting spin, formatted correctly for the model
        start = randSpin()

    # turn model to evaluate mode
    model.eval()
    with torch.no_grad():
        logProbs, hidden = model(start)
    nextSpinProb = np.exp(logProbs[-1])
    if np.random.rand() < nextSpinProb[1].item():
        nextSpin = torch.Tensor([[[1]]]).float()
    else:
        nextSpin = torch.Tensor([[[0]]]).float()
    
    return nextSpin

def generate(model, size):
    # generates entire chain of spins of specified size
    spins = randSpin()
    for i in range(size):
        newSpin = predict(model, spins).reshape(1,1,1).float()
        spins = torch.cat((spins, newSpin), dim=1)
    return spins

import datetime
def experiment(filename):
    now = datetime.datetime.now()
    with open(filename, 'a') as out:
        out.write(now.strftime("%Y-%m-%d %H:%M:%S") + "\n-------------------------------\n")
        out.write("model: " + str(model) + "\n\n")
        out.write("loss_fn: " + str(loss_fn) + "\n\n")
        out.write("optimizer: " + str(optimizer) + "\n\n")
        out.write("N_EPOCHS: " + str(N_EPOCHS) + "\n\n")
        out.write("BATCH_SIZE: ", + str(BATCH_SIZE) + "\n\n")
        out.write("dataFile: " + str(dataFile) + "\n\n")
    return
