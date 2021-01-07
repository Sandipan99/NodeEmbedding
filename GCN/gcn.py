import torch
from utils import Trainer
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class GCN_Net(nn.Module):
    def __init__(self, feature_size,embedding_size,output_size):
        super(GCN_Net, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)
        self.gcn2 = GCN(16, 7, F.relu)

    def forward(self, g, features,train=True):
        x = self.gcn1(g, features)
        if train:
            x = self.gcn2(g, x)
        return x


class GCN_Trainer(Trainer):
    def __init__(self, graph, features, labels, train_mask, test_mask, val_mask, model, epochs, path):
        super(GCN_Trainer, self).__init__(graph, features, labels, train_mask,
                                          test_mask, val_mask, model, epochs, path)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        max_score = 0
        for epoch in range(self.epochs):

            logits = self.model(self.g, self.features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'epoch - {epoch} | Loss - {loss}')

            score = self.evaluate()
            if score > max_score:
                max_score = score
                torch.save(self.model.state_dict(), self.path)  # saves model with the best validation accuracy

    def evaluate(self, val=True):
        logits = self.model(self.g, self.features)
        logp = F.log_softmax(logits, 1)
        if val:
            inf_label = logp[self.val_mask].argmax(dim=1)
            true_label = self.labels[self.val_mask]
        else:
            inf_label = logp[self.test_mask].argmax(dim=1)
            true_label = self.labels[self.test_mask]

        return (accuracy_score(true_label.numpy(), inf_label.numpy()))