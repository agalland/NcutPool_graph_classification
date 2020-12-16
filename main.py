from utils import *
from train import graphClassifier

import ssl


ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device != "cpu"
device = "cpu"
print(device)

dataset_names = ["PROTEINS", "DD", "COLLAB"]

data_path = "data/"

h_dims = [16, 16]
lr = 1e-1
numEpochs = 2000
epochEval = 1
same_size = False
verbose = True
results_path = "results/"
path_weights = "weights/"
num_pool = 1
ps = [50]
n_gcns = 1
decrease_lr = False


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
    for lr in [1e-2]:
        for alpha_reg in [0.01]:
            for ps in [[1], [70]]:
                for num_pool in [1, 2]:
                    for n_gcns in [2]:
                        for h_dims in [[16, 8], [64, 16]]:
                            print("ps: {}".format(ps))
                            print("num pool: {}".format(num_pool))
                            fname = dataset_name + str(h_dims) + str(lr) + str(alpha_reg) + str(ps[0]) + str(num_pool) + str(n_gcns) + "_val.npy"
                            net = graphClassifier
                            cross_val(net,
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
                                      decrease_lr)