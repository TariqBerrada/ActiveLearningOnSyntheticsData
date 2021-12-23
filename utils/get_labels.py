import sys
sys.path.append('.')

import numpy as np
import joblib

from utils.generate_data import b1, b2, xmin, xmax, ymin, ymax

import matplotlib.pyplot as plt

def label_pts(x):
    """generates labels for data points.

    Args:
        x ([array]): [n_pts, n_dim] n_dim = 2 for x, y coords.
    """
    labels = np.zeros((x.shape[0], 3)) # 3 is the number of classesa s we have 2 boundaries.
    
    bound1 = b1(x[:, 0]) # upper bound
    bound2 = b2(x[:, 0]) # lower bound

    idx1 = np.where(x[:, 1] > bound1)
    idx2 = np.where(x[:, 1] < bound2)

    labels[idx1, 0] = 1
    labels[idx2, 2] = 1
    labels[:, 1] = 1 - labels[:, 0] - labels[:, 2]

    return labels
if __name__ == '__main__':
    n_pts = 2000
    cloud = np.random.uniform(low=0, high = 1, size = (n_pts, 2))
    cloud[:, 0] = cloud[:, 0]*(xmax - xmin) + xmin
    cloud[:, 1] = cloud[:, 1]*(ymax - ymin) + ymin

    labels = label_pts(cloud)

    joblib.dump({'x' : cloud, 'y': labels}, 'data/dataset_init.pt')

    plt.scatter(cloud[:, 0], cloud[:, 1], c = labels, s = 2)
    plt.show()