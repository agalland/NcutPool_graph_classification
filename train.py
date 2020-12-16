import torch

import numpy as np
import torch.optim as optim
import torch.nn as nn

from sklearn.metrics import accuracy_score, roc_auc_score
from model import GAE


class graphClassifier(object):
    def __init__(self,
                 n_feat=None,
                 h_dims=None,
                 out_dim=None,
                 num_pool=None,
                 n_gcns=None,
                 ps=None,
                 path_save_weights=None,
                 alpha_reg=0.1,
                 alpha_hloss=0.,
                 lr=0.01,
                 numEpochs=100,
                 epochEval=1,
                 device="cpu"):

        self._n_feat = n_feat
        self._h_dims = h_dims
        self._out_dim = out_dim
        self._num_pool = num_pool
        self._n_gcns = n_gcns
        self.ps = ps
        self.path_save_weights = path_save_weights
        self.numEpochs = numEpochs
        self.epochEval = epochEval
        self.lr = lr
        self.device = device
        self.alpha_reg = alpha_reg
        self.alpha_hloss = alpha_hloss

        self._net = GAE(self._n_feat,
                        self._h_dims,
                        self._out_dim,
                        self.ps,
                        self._num_pool,
                        self._n_gcns).to(device)

    def fit(self, X_train, y_train, X_test, y_test, verbose=True, epoch_verbose=1, decrease_lr=False):
        label_train = y_train.astype(np.int64)
        label_train = torch.LongTensor(label_train).to(self.device)
        label_test = y_test.astype(np.int64)
        label_test = torch.LongTensor(label_test).to(self.device)

        optimizer = optim.Adam(params=self._net.parameters(),
                               lr=self.lr,
                               weight_decay=1e-5)

        self.criterion = nn.NLLLoss()

        self.loss_test_min = np.infty
        loss_test_over = 0.
        loss_test_over_nb = 50
        loss_test_prev = np.infty
        reduce_lr = True
        self.epochBreak = self.numEpochs
        for epoch in range(self.numEpochs):
            # Initialize grad
            optimizer.zero_grad()
            pred_train, loss_reg_train = self.forward(X_train,
                                                      label_train)
            pred_test, loss_reg_test = self.forward(X_test, label_test)

            loss_train = pred_train + loss_reg_train
            loss_test = pred_test + loss_reg_test
            # Loss backward
            loss_train.backward(retain_graph=True)
            optimizer.step()

            if decrease_lr:
                if torch.abs(loss_test - loss_test_prev) < 1e-5 and reduce_lr:
                    print("reduce lr")
                    self.lr /= 10.
                    optimizer = optim.Adam(params=self._net.parameters(),
                                           lr=self.lr,
                                           weight_decay=1e-3)
                    reduce_lr = False
                loss_test_prev = loss_test

            if epoch % self.epochEval == 0:
                if loss_test.data < self.loss_test_min:
                    self.loss_test_min = loss_test.data
                    self.weights = torch.save(self._net.state_dict(), self.path_save_weights)
                    loss_test_over = 0
                else:
                    loss_test_over += 1
                if loss_test_over > loss_test_over_nb:
                    self._net.load_state_dict(torch.load(self.path_save_weights))
                    print("break at epoch: {}".format(epoch - loss_test_over_nb - 1))
                    self.epochBreak = epoch
                    break

                if verbose:
                    if epoch % epoch_verbose == 0:
                        print("Epoch: {}, loss train: {}, loss test: {},".format(epoch,
                                                                                 loss_train.data,
                                                                                 loss_test.data))

    def forward(self, X, y):
        pred = 0.
        loss_reg_att = 0.
        for i, (adj, feature) in enumerate(X):
            adj_tensor = torch.Tensor(adj.todense()).to(self.device)
            adjC_tensor = self.normalize_adj(adj_tensor, 2)
            feature_tensor = torch.Tensor(feature.todense()).to(self.device)
            output, Cs, atts = self._net(feature_tensor, adjC_tensor, adj_tensor)
            for k in range(len(atts)):
                #Cut minimization
                if self.alpha_reg > 0.:
                    mat = torch.matmul(torch.matmul(Cs[k], atts[k]), Cs[k].t())
                    ind = np.diag_indices(mat.shape[0])
                    vol = mat[ind[0], ind[1]]
                    mat[ind[0], ind[1]] = 0.
                    loss_reg_att += self.alpha_reg * (mat.sum(1) / vol).sum()
            output = output.view(1, -1)
            pred += self.criterion(output, y[i].unsqueeze(0))

        pred /= len(X)
        loss_reg_att /= len(X)

        return pred, loss_reg_att

    def evaluate(self, X, y):
        pred = None
        for i, (adj, feature) in enumerate(X):
            adj_tensor = torch.Tensor(adj.todense()).to(self.device)
            adjC_tensor = self.normalize_adj(adj_tensor, 2)
            feature_tensor = torch.Tensor(feature.todense()).to(self.device)
            output, _, _ = self._net(feature_tensor, adjC_tensor, adj_tensor)
            output = output.view(1, -1)
            if pred is None:
                pred = output
            else:
                pred = torch.cat((pred, output), 0)
        pred_soft = pred.cpu().data.numpy()
        valpred, pred = torch.max(pred, 1)
        pred = pred.cpu().data.numpy()

        y_multi = np.zeros((len(y), len(np.unique(y))))
        y_multi[range(len(y)), y] = 1.
        acc = np.round(accuracy_score(y, pred), 3)
        roc = np.round(roc_auc_score(y_multi, pred_soft), 3)

        return acc, roc

    def HLoss(self, prob):
        b = -prob * torch.log(prob + 1e-10)
        b = torch.sum(b, 1)
        b = torch.mean(b)

        return b

    def normalize_adj(self, adj, k):
        adj = adj + k * torch.eye(adj.size(0))
        D = torch.sqrt(adj.sum(0))
        adj = adj / D
        adj = adj.transpose(0, 1)
        adj = adj / D
        adj = adj.transpose(0, 1)

        return adj