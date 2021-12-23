import joblib, tqdm, torch

import numpy as np

from model.classifier import Classifier
from utils.trainer import train
from utils.dataloader import split

from utils.generate_data import xmin, xmax, ymin, ymax, b1, b2

import matplotlib.pyplot as plt
plt.style.use('bmh')

data = joblib.load('data/dataset_init.pt')
data_train, data_test = split(data, perc = .75)

model = Classifier(input_dim = 2, output_dim = 3).to('cpu')
model.device = 'cpu'
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
criterion = torch.nn.CrossEntropyLoss(reduction='none')

train(model, data_train, data_test, optimizer, criterion, n_epochs = 60)
