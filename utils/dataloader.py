from torch.utils.data import Dataset

class DatasetClass(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return self.data['x'].shape[0]
    
    def __getitem__(self, idx):
        return {'x': self.data['x'][idx, :], 'y': self.data['y'][idx, :]}

def split(data, perc):
    n = data['x'].shape[0]
    sep = int(n*perc)

    data_train = {'x': data['x'][:sep, :], 'y': data['y'][:sep, :]}
    data_test = {'x': data['x'][sep:, :], 'y': data['y'][sep:, :]}
    
    return data_train, data_test