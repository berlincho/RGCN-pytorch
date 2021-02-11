from __future__ import print_function

import scipy.sparse as sp
import numpy as np


def csr_zero_rows(csr, rows_to_zero):
    """Set rows given by rows_to_zero in a sparse csr matrix to zero.
    NOTE: Inplace operation! Does not return a copy of sparse matrix."""
    rows, cols = csr.shape
    mask = np.ones((rows,), dtype=np.bool)
    mask[rows_to_zero] = False
    nnz_per_row = np.diff(csr.indptr)

    mask = np.repeat(mask, nnz_per_row)
    nnz_per_row[rows_to_zero] = 0
    csr.data = csr.data[mask]
    csr.indices = csr.indices[mask]
    csr.indptr[1:] = np.cumsum(nnz_per_row)
    csr.eliminate_zeros()
    return csr


def csc_zero_cols(csc, cols_to_zero):
    """Set rows given by cols_to_zero in a sparse csc matrix to zero.
    NOTE: Inplace operation! Does not return a copy of sparse matrix."""
    rows, cols = csc.shape
    mask = np.ones((cols,), dtype=np.bool)
    mask[cols_to_zero] = False
    nnz_per_row = np.diff(csc.indptr)

    mask = np.repeat(mask, nnz_per_row)
    nnz_per_row[cols_to_zero] = 0
    csc.data = csc.data[mask]
    csc.indices = csc.indices[mask]
    csc.indptr[1:] = np.cumsum(nnz_per_row)
    csc.eliminate_zeros()
    return csc


def sp_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (dim, 1)
    data = np.ones(len(idx_list))
    row_ind = list(idx_list)
    col_ind = np.zeros(len(idx_list))
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors."""
    sp_nodes = sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(sp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def bfs(adj, roots):
    """
    Perform BFS on a graph given by an adjaceny matrix adj.
    Can take a set of multiple root nodes.
    Root nodes have level 0, first-order neighors have level 1, and so on.]
    """
    visited = set()
    current_lvl = set(roots)
    while current_lvl:
        for v in current_lvl:
            visited.add(v)

        next_lvl = get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference
        yield next_lvl

        current_lvl = next_lvl


def bfs_relational(adj_list, roots):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = list()
    for rel in range(len(adj_list)):
        next_lvl.append(set())

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        for rel in range(len(adj_list)):
            next_lvl[rel] = get_neighbors(adj_list[rel], current_lvl)
            next_lvl[rel] -= visited  # set difference

        yield next_lvl

        current_lvl = set.union(*next_lvl)


def bfs_sample(adj, roots, max_lvl_size):
    """
    BFS with node dropout. Only keeps random subset of nodes per level up to max_lvl_size.
    'roots' should be a mini-batch of nodes (set of node indices).

    NOTE: In this implementation, not every node in the mini-batch is guaranteed to have
    the same number of neighbors, as we're sampling for the whole batch at the same time.
    """
    visited = set(roots)
    current_lvl = set(roots)
    while current_lvl:

        next_lvl = get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        for v in next_lvl:
            visited.add(v)

        yield next_lvl

        current_lvl = next_lvl


def get_splits(y, train_idx, test_idx, validation=True):
    # Make dataset splits
    # np.random.shuffle(train_idx)
    if validation:
        val_num = int(len(train_idx) // 5)
        idx_train = train_idx[val_num:]
        idx_val = train_idx[:val_num]
        idx_test = test_idx  # report final score on validation set for hyperparameter optimization
    
    else:
        idx_train = train_idx
        idx_val = train_idx  # no validation
        idx_test = test_idx

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)

    y_train[idx_train] = np.array(y[idx_train].todense())
    y_val[idx_val] = np.array(y[idx_val].todense())
    y_test[idx_test] = np.array(y[idx_test].todense())

    return y_train, y_val, y_test, idx_train, idx_val, idx_test


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten())
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten())
        a_norm = d.dot(adj).tocsr()
    return a_norm


def sym_normalize(AA):
    """Symmetrically normalize adjacency matrix."""
    A_list = list()
    for i in range(len(AA)):
        adj = sp.coo_matrix(AA[i])
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        A_list.append(AA[i].dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo())
    return A_list

def row_normalize(AA):
    A_list = list()
    for i in range(len(AA)):
        d = np.array(AA[i].sum(1)).flatten()
        d_inv = 1. / d
        d_inv[np.isinf(d_inv)] = 0.
        D_inv = sp.diags(d_inv)
        A_list.append(D_inv.dot(AA[i]).tocsr())
    return A_list

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)