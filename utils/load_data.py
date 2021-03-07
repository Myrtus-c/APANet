import pickle
import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from utils.normalization import chebynet_laplacian, bernstein_laplacian
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_networkx

data_dir = os.path.expanduser('~') + '/AnomalyGraphNet/data/'

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(dataset, split='0.5_0.2_0.3', normalization='Cheb', fixed_split=True):
    graph, x, y = load_raw_dataset(dataset)
    adj = nx.adj_matrix(graph)

    if normalization == 'Cheb':
        laplacian = chebynet_laplacian(adj)
    elif normalization == 'Bern':
        laplacian = bernstein_laplacian(adj)
    else:
        laplacian = None

    laplacian = sparse_mx_to_torch_sparse_tensor(laplacian)


    if fixed_split:
        with open(data_dir+dataset+split+'.split', 'rb') as f:
            split_dict = pickle.load(f)
            if len(split.split('_')) == 3:
                idx_train, idx_val, idx_test = split_dict['train'], split_dict['val'], split_dict['test']
            else:
                idx_train, idx_test = split_dict['train'], split_dict['test']

    else:
        train_size, val_size, test_size = [float(i) for i in split.split("_")]
        idx_train, idx_val, idx_test = \
            train_val_test_split(np.arange(len(y)), train_size=train_size,
                                 val_size=val_size, test_size=test_size,
                                 stratify=y, random_state=np.random.randint(1000))

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        # split_dict = {'train': idx_train, 'val': idx_val, 'test': idx_test}
        # pickle.dump(split_dict ,open(data_dir+dataset+split+'.split', 'wb'))


    if len(split.split('_')) == 3:
        return laplacian, x, y, (idx_train, idx_val, idx_test)
    else:
        return laplacian, x, y, (idx_train, idx_test)



def load_pyg_data(dataset, split='0.5_0.2_0.3', fixed_split=True):
    graph, x, y = load_raw_dataset(dataset)
    data = from_networkx(graph)
    data.x = x
    data.y = y

    if fixed_split:
        with open(data_dir+dataset+split+'.split', 'rb') as f:
            split_dict = pickle.load(f)
            if len(split.split('_')) == 3:
                idx_train, idx_val, idx_test = split_dict['train'], split_dict['val'], split_dict['test']
            else:
                idx_train, idx_test = split_dict['train'], split_dict['test']
    else:
        train_size, val_size, test_size = [float(i) for i in split.split("_")]
        idx_train, idx_val, idx_test = \
            train_val_test_split(np.arange(len(y)), train_size=train_size,
                                 val_size=val_size, test_size=test_size,
                                 stratify=y, random_state=np.random.randint(1000))

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    if len(split.split('_')) == 3:
        data.train_mask = idx_train
        data.val_mask = idx_val
        data.test_mask = idx_test
    else:
        data.train_mask = idx_train
        data.test_mask = idx_test

    return data



def load_raw_dataset(dataset):
    """
    :param dataset:
    :return:   graph: nx.Digraph
               x: torch.tensor
               y: torch.tensor
    """
    with open(data_dir+dataset+'.graph', 'rb') as f:
        graph = pickle.load(f)
    with open(data_dir+dataset+'.x', 'rb') as f:
        x = pickle.load(f)
    with open(data_dir+dataset+'.y', 'rb') as f:
        y = pickle.load(f)
    return graph, x, y


def train_val_test_split(*arrays,
                         train_size=0.5,
                         val_size=0.3,
                         test_size=0.2,
                         stratify=None,
                         random_state=None):

    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result


if __name__ == '__main__':

    o = load_data("chinatel", fixed_split=False, split="0.5_0.2_0.3")
    debug = 1
