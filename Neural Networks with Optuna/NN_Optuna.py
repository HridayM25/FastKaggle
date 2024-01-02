import numpy as np 
import pandas as pd
import torch 
from sklearn.model_selection import train_test_split 
import xgboost 
import lightgbm
import catboost
from torch import nn
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
import os
from sklearn.preprocessing import LabelEncoder
import warnings
import torch.nn.functional as F
import optuna 
import matplotlib.pyplot as plt
import torch.optim as optim

#Give X,y based on the dataset, using dummy here: 

X = np.random.randn(1000,10)
y = np.random.randint(0,2,1000)

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()


X_train, X_val, y_train , y_val = train_test_split(X,y, test_size=0.3)

INPUT_LAYER = X_train.shape[1]

class NeuralNetwork(nn.Module):
    def __init__(self, l1=32, l2=8, l3=16):
        super(NeuralNetwork,self).__init__()
        self.fc1 = nn.Linear(INPUT_LAYER, l1)
        self.fc2 = nn.Linear (l1, l2)
        self.fc3 = nn.Linear (l2, l3)
        self.fc4 = nn.Linear (l3, 3)
    
    def forward (self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
def objective(trial):
    l1 = trial.suggest_int('l1', 2, 256)
    l2 = trial.suggest_int('l2', 2, 256)
    l3 = trial.suggest_int('l3', 2, 256)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    model = NeuralNetwork(l1=l1, l2=l2, l3=l3)
    crit = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1000):
        outputs = model(X_train)
        loss = crit(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        output = model(X_val)
        valid_loss = crit(output, y_val)
        print(valid_loss)
    return valid_loss.item()

run = 0 # Set for no optuna

if run ==1:
    study = optuna.create_study(direction='minimize') 
    study.optimize(objective, n_trials=20)
    
    print(study.best_params) # Will give the best parameters
    
"""
Set Model based on the best parameters given by optuna

class OptimizedNeuralNetwork(nn.Module):
    def __init__(self, l1=32, l2=16, l3=8):
        super(OptimizedNeuralNetwork,self).__init__()
        self.fc1 = nn.Linear(INPUT_LAYER, l1)
        self.fc2 = nn.Linear (l1, l2)
        self.fc3 = nn.Linear (l2, l3)
        self.fc4 = nn.Linear (l3, 3)
    
    def forward (self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
    
model = OptimizedNeuralNetwork()        
crit = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.004695479567227815)

for epoch in range(3000):
    outputs = model(X)
    loss = crit(outputs, y)
    print(f"The loss at {epoch} Epoch is -----> {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
Train the neural network with the best parameters given by optuna, and then fit on the whole dataset.

Then use the trained model to predict on the test set.

"""