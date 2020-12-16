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
alpha_reg = 0.001
results_path = "results/"
path_weights = "weights/"

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