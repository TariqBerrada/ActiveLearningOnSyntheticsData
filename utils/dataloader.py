from torch.utils.data import Dataset
import torch

from utils.get_labels import label_pts

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

def augment_until(data, n):
    # augment data with datapoints at the boundary of data until having n samples.
    # print('before augmentation', data['x'].shape)
    nsamples = data['x'].shape[0]
    if nsamples >= n:
        return data
    else:
        remaining = n - nsamples
        neighbors = torch.randint(0, nsamples-1, (remaining,))
        new = torch.normal(0, .2, size = (remaining, 2), dtype = torch.float32, device=data['x'].device)
        data_to_add = data['x'][neighbors] + new
        labels_n = label_pts(data_to_add.cpu().numpy())
        labels_n = torch.tensor(labels_n, dtype = torch.float32, device = data['x'].device)

        data['x'] = torch.cat((data['x'], data_to_add))
        data['y'] = torch.cat((data['y'], labels_n))
    # print('after augmentation', data['x'].shape)
    return data
