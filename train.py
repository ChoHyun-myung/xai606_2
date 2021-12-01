from typing import Union, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import numpy as np
import pandas as pd
import scipy.sparse

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, init_features):
        super().__init__()
        self.l1 = GCNConv(num_node_features, init_features)
        self.l2 = GCNConv(init_features, init_features * 2)
        self.l3 = GCNConv(init_features * 2, init_features * 4)
        self.l4 = GCNConv(init_features * 4, init_features * 8)
        self.l5 = GCNConv(init_features * 8, init_features * 4)
        self.l6 = GCNConv(init_features * 4, init_features * 2)
        self.l7 = GCNConv(init_features * 2, init_features)

        self.out = torch.nn.Linear(init_features, 1)

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index

        x = self.l1(x, edge_index)
        x = F.relu(x)
        x = self.l2(x, edge_index)
        x = F.relu(x)
        x = self.l3(x, edge_index)
        x = F.relu(x)
        x = self.l4(x, edge_index)
        x = F.relu(x)
        x = self.l5(x, edge_index)
        x = F.relu(x)
        x = self.l6(x, edge_index)
        x = F.relu(x)
        x = self.l7(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch=batch)

        out = self.out(x)
        out = torch.squeeze(out)

        return out


class GraphDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.data = pd.read_csv('data/' + dataset + '.csv')

        encoder = LabelEncoder()

        self.data['model'] = encoder.fit_transform(self.data['model'])
        self.data['transmission'] = encoder.fit_transform(self.data['transmission'])
        self.data['fuelType'] = encoder.fit_transform(self.data['fuelType'])

        self.x = self.data.drop('price', axis=1)
        self.y = self.data['price']

        scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
        self.x = scaler.fit_transform(self.x)

    def __getitem__(self, idx):
        graph = self.convert2graph(self.x[idx], self.y[idx])
        graph = graph.to(torch.device('cuda'))
        return graph

    def __len__(self):
        return len(self.x)

    @staticmethod
    def convert2graph(x, y):
        """
        x = [model, year, transmission, mileage, fuelType, tax, mpg, engineSize, price]
        :param x:
        :param y:
        :return:
        """

        # adj = np.ones([len(x), len(x)])
        # np.fill_diagonal(adj, 0)
        # adj = scipy.sparse.csr_matrix(adj)      # Make it to scipy.sparse matrix
        # adj = adj.tocoo()
        # row = torch.from_numpy(adj.row).to(torch.long)
        # col = torch.from_numpy(adj.col).to(torch.long)
        # edge_index = torch.stack([row, col], dim=0)
        # x = [[model, year], [trans, fuel], [mile, mpg], [tax, eng]]
        x = torch.tensor([
            [x[0], x[1]],
            [x[2], x[4]],
            [x[3], x[6]],
            [x[5], x[7]]
        ], dtype=torch.float)
        edge_idx = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2],
                                 [1, 0, 2, 0, 3, 0, 2, 1]], dtype=torch.long)
        graph = Data(x=x, edge_index=edge_idx, y=y)

        return graph


def train_fn(tr_loader, valid_loader, test_loader, epochs):
    model = GCN(num_node_features=2, init_features=256).to(torch.device('cuda'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        loss_val = 0
        for data in tr_loader:
            optimizer.zero_grad()

            out = model(data, data.batch)
            y = torch.tensor(data.y, dtype=torch.float, device='cuda', requires_grad=True)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            loss_val += np.sqrt(loss.item())
        loss_val /= len(tr_loader)
        print(f'Epoch [{epoch+1}] train loss = {loss_val:.5f}')

        loss_val = validation(model, valid_loader, loss_fn)

        print(f'Epoch [{epoch+1}] validation loss = {loss_val:.5f}')

        loss_val = validation(model, test_loader, loss_fn)

        print(f'Epoch [{epoch+1}] test loss = {loss_val:.5f}')


def validation(model, loader, loss_fn):
    model.eval()

    loss_val = 0
    for data in loader:
        out = model(data, data.batch)
        y = torch.tensor(data.y, dtype=torch.float, device='cuda', requires_grad=False)
        loss = loss_fn(out, y)
        loss_val += np.sqrt(loss.item())
    loss_val /= len(loader)

    return loss_val


if __name__ == '__main__':
    batch_size = 128
    train_data = GraphDataset('train')
    val_data = GraphDataset('val')
    test_data = GraphDataset('test')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    train_fn(train_loader, valid_loader, test_loader, epochs=100)
