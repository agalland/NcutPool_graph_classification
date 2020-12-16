import torch
import sys

import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pickle as pkl

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
from ogb.graphproppred import GraphPropPredDataset


def flatten(input):
    return input.view(-1)

def unflatten(input, size0, size1):
    return input.view(size0, size1)

def training_set_molecules(dataset):
    X = []
    y = []
    size_max = dataset.max_num_nodes
    for i in range(dataset.n_graphs_):
        adj = nx.adjacency_matrix(dataset.graphs_[i])
        # adj = normalize(adj + sp.eye(adj.shape[0]), norm="l1", axis=1)
        nodeFeature = sp.csr_matrix(dataset.node_features[i])

        if adj.shape[0] != nodeFeature.shape[0]:
            continue
        # adj_i = sp.csr_matrix(np.zeros((size_max, size_max)), shape=(size_max, size_max))
        #
        # feature_i = sp.csr_matrix(np.zeros((size_max, nodeFeature.shape[1])), shape=(size_max, nodeFeature.shape[1]))
        # adj_i[:adj.shape[0], :adj.shape[0]] = adj
        # feature_i[:adj.shape[0]] = nodeFeature
        # adj_i = sp.csr_matrix(adj_i)
        # feature_i = sp.csr_matrix(feature_i)
        X.append((adj, nodeFeature))
        y.append(dataset.labels_[i])

    return X, y

class GraphDataset:
    def __init__(self, folder_path=''):
        if folder_path.split("-")[0] != "ogbg":
            g = nx.Graph()
            data_adj = np.loadtxt(folder_path + '_A.txt', delimiter=',').astype(int)
            data_graph_indicator = np.loadtxt(folder_path + '_graph_indicator.txt',
                                              delimiter=',').astype(int)
            labels = np.loadtxt(folder_path + '_graph_labels.txt',
                                delimiter=',').astype(int)
            # If features aren't available, compute one-hot degree vectors
            try:
                node_labels = np.loadtxt(folder_path + '_node_labels.txt', delimiter=',').astype(int)
                node_labels -= np.min(node_labels)
                max_feat = np.max(node_labels) + 1
                mat_feat = np.eye(max_feat)
                with_node_features = True
            except:
                with_node_features = False

            data_tuple = list(map(tuple, data_adj))
            g.add_edges_from(data_tuple)
            g.remove_nodes_from(list(nx.isolates(g)))

            le = LabelEncoder()
            self.labels_ = le.fit_transform(labels)
            self.n_classes_ = len(le.classes_)
            self.n_graphs_ = len(self.labels_)

            graph_num = data_graph_indicator.max()
            node_list = np.arange(data_graph_indicator.shape[0]) + 1
            self.graphs_ = []
            self.node_features = []
            max_num_nodes = 0
            self.degree_max = 0
            for i in range(graph_num):
                if i % 500 == 0:
                    print("{}%".format(round((i * 100) / graph_num), 3))

                nodes = node_list[data_graph_indicator == i + 1]
                g_sub = g.subgraph(nodes).copy()

                max_cc = max(nx.connected_components(g_sub), key=len)
                g_sub = g_sub.subgraph(max_cc).copy()

                adj = np.array(nx.adjacency_matrix(g_sub).todense())
                self.degree_max = max(self.degree_max, np.max(np.sum(adj, 0)))
                nodes = range(len(adj))
                g_sub.graph['label'] = self.labels_[i]
                nx.convert_node_labels_to_integers(g_sub)

                tmp = len(nodes)
                self.graphs_.append(g_sub)
                if tmp > max_num_nodes:
                    max_num_nodes = tmp

                if with_node_features:
                    nodes = list(g_sub.nodes()) - np.min(list(g.nodes()))
                    feat_index = node_labels[nodes]
                    node_feat = mat_feat[feat_index]
                    self.node_features.append(node_feat)

            if not with_node_features:
                mat_feat = np.eye(self.degree_max+1)
                for i in range(graph_num):
                    g_sub = self.graphs_[i]
                    deg = np.array(list(dict(nx.degree(g_sub)).values()))
                    node_feat = mat_feat[deg]
                    self.node_features.append(node_feat)
        else:
            dataset = GraphPropPredDataset(name=folder_path)
            split_idx = dataset.get_idx_split()
            train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
            self.train_idx = train_idx
            self.test_idx = test_idx
            self.valid_idx = valid_idx
            self.labels_ = dataset.labels.reshape(-1)
            self.graphs_ = []
            self.node_features = []
            max_feat, min_feat = get_boundaries_features(dataset)
            max_num_nodes = 0.
            for k in range(len(dataset.graphs)):
                edge_idx = dataset.graphs[k]["edge_index"]
                a = np.zeros((dataset.graphs[k]["num_nodes"],
                              dataset.graphs[k]["num_nodes"]))
                a[edge_idx[0], edge_idx[1]] = 1.
                g = nx.from_numpy_matrix(a)
                self.graphs_.append(g)
                graph_feat = get_one_hot_feature(max_feat,
                                                 min_feat,
                                                 dataset.graphs[k]["node_feat"])
                # graph_feat = dataset.graphs[k]["node_feat"]
                self.node_features.append(graph_feat)
                if dataset.graphs[k]["num_nodes"] > max_num_nodes:
                    max_num_nodes = dataset.graphs[k]["num_nodes"]

        self.n_graphs_ = len(self.graphs_)
        self.graphs_ = np.array(self.graphs_)
        self.max_num_nodes = max_num_nodes

        print('Loaded {} graphs,\
              the max number of nodes is {}'.format(self.n_graphs_,
                                                    self.max_num_nodes))


def get_boundaries_features(dataset):
    max_feat = - np.ones(9) * np.inf
    min_feat = np.ones(9) * np.inf
    for k in range(len(dataset.graphs)):
        node_feat = dataset.graphs[k]["node_feat"]
        node_max_feat = np.max(node_feat, 0)
        node_min_feat = np.min(node_feat, 0)
        max_feat = np.concatenate(
            (max_feat.reshape(-1, 1), node_max_feat.reshape(-1, 1)), 1)
        min_feat = np.concatenate(
            (min_feat.reshape(-1, 1), node_min_feat.reshape(-1, 1)), 1)
        max_feat = np.max(max_feat, 1)
        min_feat = np.min(min_feat, 1)
        max_feat = max_feat.reshape(-1)
        min_feat = min_feat.reshape(-1)
    return max_feat, min_feat


def get_one_hot_feature(max_feat, min_feat, node_feat):
    range_feat = max_feat - min_feat + 1
    mat_feat = []
    for k in range(len(range_feat)):
        mat_feat.append(np.eye(int(range_feat[k]) + 1))
    graph_feat_one_hot = None
    for node in range(node_feat.shape[0]):
        node_feat_one_hot = None
        for f in range(node_feat.shape[1]):
            if len(mat_feat[f]) == 0:
                continue
            if node_feat_one_hot is None:
                node_feat_one_hot = mat_feat[f][node_feat[node][f]]
            else:
                node_feat_one_hot = np.concatenate(
                    (node_feat_one_hot, mat_feat[f][node_feat[node][f]]))
        if graph_feat_one_hot is None:
            graph_feat_one_hot = node_feat_one_hot.reshape(1, -1)
        else:
            graph_feat_one_hot = np.concatenate(
                (graph_feat_one_hot, node_feat_one_hot.reshape(1, -1)), 0)

    return graph_feat_one_hot


def get_degree_feature(G):
    deg = dict(nx.degree(G))
    degs_sorted = sorted(list(np.unique(list(deg.values()))))
    pd_degs = pd.DataFrame(index=degs_sorted, columns=["num"])
    pd_degs["num"] = range(len(degs_sorted))
    pd_degs_inverse = pd.DataFrame(index=list(pd_degs["num"]), columns=["deg"])
    pd_degs_inverse["deg"] = list(pd_degs.index)
    num_degs = len(np.unique(list(deg.values())))
    mat = np.eye(num_degs)
    degs = list(deg.values())
    id_degs = np.array(list(pd_degs.loc[degs, "num"]))
    node_features = mat[id_degs]
    node_features = sp.csr_matrix(node_features)

    return node_features


def normalize_adj(dataset,
                  normalize_bool=True):
    X = []
    y = []
    for i in range(dataset.n_graphs_):
        adj = nx.adjacency_matrix(dataset.graphs_[i])
        if normalize_bool:
            adj = np.array(normalize(adj + 2 * sp.eye(adj.shape[0])))
            adj = sp.csr_matrix(adj)
        nodeFeature = np.array(dataset.node_features[i])
        nodeFeature = sp.csr_matrix(nodeFeature)

        if adj.shape[0] != nodeFeature.shape[0]:
            continue

        X.append((adj, nodeFeature))
        y.append(dataset.labels_[i])

    y = np.array(y)

    return X, y


def training_set_molecules(dataset, same_size=False):
    X = []
    y = []
    size_max = dataset.max_num_nodes
    for i in range(dataset.n_graphs_):
        adj = nx.adjacency_matrix(dataset.graphs_[i])
        adj = normalize_adj(adj, 1)
        nodeFeature = np.array(dataset.node_features[i])
        nodeFeature = sp.csr_matrix(nodeFeature)

        if adj.shape[0] != nodeFeature.shape[0]:
            continue
        adj_i = sp.lil_matrix(np.zeros((size_max, size_max)), shape=(size_max, size_max))

        feature_i = sp.lil_matrix(np.zeros((size_max, nodeFeature.shape[1])),
                                  shape=(size_max, nodeFeature.shape[1]))
        adj_i[:adj.shape[0], :adj.shape[0]] = adj
        feature_i[:adj.shape[0]] = nodeFeature
        adj_i = sp.lil_matrix(adj_i)
        feature_i = sp.lil_matrix(feature_i)
        if same_size:
            X.append((adj_i, feature_i))
            y.append(dataset.labels_[i])
        else:
            X.append((adj, nodeFeature))
            y.append(dataset.labels_[i])

    y = np.array(y)

    return X, y


def cross_val(net,
              X,
              y,
              dataset_name,
              results_path,
              n_feat,
              h_dims,
              out_dim,
              num_pool,
              n_gcns,
              alpha_reg,
              ps,
              lr,
              numEpochs,
              epochEval,
              path_weights,
              device,
              verbose,
              decrease_lr):

    path_save_weights = path_weights + dataset_name + str(h_dims) + str(lr) + str(alpha_reg) + str(ps[0]) + str(num_pool) + str(n_gcns)

    n_splits_val = 10
    skf_val = StratifiedKFold(n_splits=n_splits_val,
                              shuffle=True,
                              random_state=2)
    X_vec = np.zeros(len(X))

    accuracy_val = []
    roc_auc_val = []
    accuracy_test = []
    roc_auc_test = []

    for ifold, (train_ind1, val_ind) in enumerate(skf_val.split(X_vec, y)):
        accuracies = train_fold(X,
                                y,
                                train_ind1,
                                val_ind,
                                net,
                                ifold,
                                n_feat=n_feat,
                                h_dims=h_dims,
                                num_pool=num_pool,
                                n_gcns=n_gcns,
                                out_dim=out_dim,
                                ps=ps,
                                lr=lr,
                                numEpochs=numEpochs,
                                epochEval=epochEval,
                                path_save_weights=path_save_weights,
                                device=device,
                                verbose=verbose,
                                alpha_reg=alpha_reg,
                                decrease_lr=decrease_lr)
        accuracy_val.append(accuracies[0])
        roc_auc_val.append(accuracies[1])
        accuracy_test.append(accuracies[2])
        roc_auc_test.append(accuracies[3])

        np.save(results_path + dataset_name + str(h_dims) + str(lr) + str(
            alpha_reg) + str(ps[0]) + str(num_pool) + str(
            n_gcns) + "_val.npy",
                accuracy_val)
        np.save(results_path + dataset_name + str(h_dims) + str(lr) + str(
            alpha_reg) + str(ps[0]) + str(num_pool) + str(
            n_gcns) + "_test.npy",
                accuracy_test)

def train_fold(X,
               y,
               train_ind1,
               val_ind,
               net,
               ifold,
               n_feat,
               h_dims,
               num_pool,
               ps,
               n_gcns,
               out_dim,
               lr,
               numEpochs,
               epochEval,
               path_save_weights,
               device,
               verbose,
               alpha_reg,
               decrease_lr):

    X_val = [X[k] for k in val_ind]
    y_val = y[val_ind]

    n_splits = 9
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=0)
    X_vec = np.zeros((len(train_ind1), 2))
    y1 = y[train_ind1]

    split = skf.split(X_vec, y1)
    train_ind2, test_ind = list(split)[0]

    train_ind = train_ind1[train_ind2]
    test_ind = train_ind1[test_ind]

    X_train = [X[k] for k in train_ind]
    y_train = y[train_ind]
    X_test = [X[k] for k in test_ind]
    y_test = y[test_ind]

    path_save_weights = path_save_weights + str(ifold)
    train_net = net(n_feat=n_feat,
                    h_dims=h_dims,
                    out_dim=out_dim,
                    num_pool=num_pool,
                    n_gcns=n_gcns,
                    alpha_reg=alpha_reg,
                    ps=ps,
                    path_save_weights=path_save_weights,
                    lr=lr,
                    numEpochs=numEpochs,
                    epochEval=epochEval,
                    device=device)

    train_net.fit(X_train,
                  y_train,
                  X_test,
                  y_test,
                  verbose,
                  decrease_lr=decrease_lr)

    train_net._net.load_state_dict(torch.load(path_save_weights))

    acc_val, auc_val = train_net.evaluate(X_val, y_val)
    acc_test, auc_test = train_net.evaluate(X_test, y_test)

    np.save("weightsEval/train_ind" + str(ifold) + ".npy", train_ind)
    np.save("weightsEval/test_ind" + str(ifold) + ".npy", test_ind)
    np.save("weightsEval/val_ind" + str(ifold) + ".npy", val_ind)

    return acc_val, auc_val, acc_test, auc_test


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../../data/node_classification/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../../data/node_classification/data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + 1. * sp.eye(adj.shape[0])
    D = np.diag(np.sqrt(np.array(adj.sum(0)).squeeze(0)))
    D = sp.csr_matrix(np.linalg.inv(D))
    adj = np.dot(D, adj)
    adj = np.dot(adj, D)
    adj = sp.csr_matrix(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
