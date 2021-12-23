import torch
import torch.nn.functional as F

class Classifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 64):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.softmax(self.out(x))