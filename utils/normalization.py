import numpy as np
import scipy.sparse as sp

def sym_normalized_laplacian(adj_m):
   """
   :param adj_m:   nx.adj_matrix
   :return: I - D^(-1/2)AD^(-1/2)  scipy sparse tensor
   """
   adj_m = sp.coo_matrix(adj_m)
   row_sum = np.array(adj_m.sum(1))
   d_mat = sp.diags(row_sum.flatten())
   L = d_mat - adj_m
   d_inv = np.power(row_sum, -1/2).flatten()
   d_inv[np.isinf(d_inv)] = 0.
   d_mat_inv = sp.diags(d_inv)
   return d_mat_inv.dot(L).dot(d_mat_inv).tocoo()


def chebynet_laplacian(adj_m, l_max=2.0):
   """
   :param adj_m:   nx.adj_matrix
   :param l_max:   lambda_max
   :return:   scipy sparse tensor
   """
   adj_m = sp.coo_matrix(adj_m)
   row_sum = np.array(adj_m.sum(1))
   d_mat = sp.diags(row_sum.flatten())
   L = d_mat - adj_m
   d_inv = np.power(row_sum, -1/2).flatten()
   d_inv[np.isinf(d_inv)] = 0.
   d_mat_inv = sp.diags(d_inv)

   return 2. / l_max * d_mat_inv.dot(L).dot(d_mat_inv).tocoo() - sp.eye(adj_m.shape[0])


def bernstein_laplacian(adj_m, l_max=2.0):
   """
   :param adj_m:   nx.adj_matrix
   :param l_max:   lambda_max
   :return:   L / lambda_max
   """
   adj_m = sp.coo_matrix(adj_m)
   row_sum = np.array(adj_m.sum(1))
   d_mat = sp.diags(row_sum.flatten())
   L = d_mat - adj_m
   d_inv = np.power(row_sum, -1/2).flatten()
   d_inv[np.isinf(d_inv)] = 0.
   d_mat_inv = sp.diags(d_inv)

   return 1. / l_max * d_mat_inv.dot(L).dot(d_mat_inv).tocoo()


# def chebynet_laplacian(adj_m, l_max=2.0):
#    """
#    :param adj_m:   nx.adj_matrix
#    :param l_max:   lambda_max
#    :return:   scipy sparse tensor
#    """
#    adj_m = sp.coo_matrix(adj_m)
#    row_sum = np.array(adj_m.sum(1))
#    d_inv = np.power(row_sum, -1/2).flatten()
#    d_inv[np.isinf(d_inv)] = 0.
#    d_mat_inv = sp.diags(d_inv)
#    return -2 / l_max * d_mat_inv.dot(adj_m).dot(d_mat_inv).tocoo() + (2 - l_max) / l_max * sp.eye(adj_m.shape[0])