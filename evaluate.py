from utils import *
from train import graphClassifier

import ssl


ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device != "cpu"
device = "cpu"
print(device)

dataset_names = ["PROTEINS"]

data_path = "data/"

h_dims = [16, 8]
lr = 1e-2
numEpochs = 2000
epochEval = 1
same_size = False
verbose = True
num_pool = 1
ps = [1]
n_gcns = 1
decrease_lr = False
load_weights = "weightsEval/"
alpha_reg = 1e-3

for dataset_name in dataset_names:
    dataset_path = data_path + dataset_name + "/" + dataset_name
    dataset = GraphDataset(dataset_path)

    X, y = normalize_adj(dataset, normalize_bool=False)
    out_dim = len(np.unique(dataset.labels_))
    adj, feature = X[0]
    n_feat = feature.shape[1]
    print("feature size: {}".format(n_feat))
    g_size = feature.shape[0]
    out_dim = int(np.max(y)) + 1

    train_ind = np.load(load_weights + "train_ind.npy")
    test_ind = np.load(load_weights + "test_ind.npy")
    val_ind = np.load(load_weights + "val_ind.npy")

    X_val = [X[k] for k in val_ind]
    y_val = y[val_ind]
    X_train = [X[k] for k in train_ind]
    y_train = y[train_ind]
    X_test = [X[k] for k in test_ind]
    y_test = y[test_ind]

    net = graphClassifier
    train_net = net(n_feat=n_feat,
                    h_dims=h_dims,
                    out_dim=out_dim,
                    num_pool=num_pool,
                    n_gcns=n_gcns,
                    alpha_reg=alpha_reg,
                    ps=ps,
                    lr=lr,
                    numEpochs=numEpochs,
                    epochEval=epochEval,
                    device=device)

    weights2load = load_weights + "weights"
    train_net._net.load_state_dict(torch.load(weights2load))

    acc_val, _ = train_net.evaluate(X_val, y_val)
    acc_test, _= train_net.evaluate(X_test, y_test)
    acc_train, _ = train_net.evaluate(X_train, y_train)

    print("accuracies: train {}, test {}, val {}".format(acc_train, acc_test, acc_val))