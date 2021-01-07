import torch
from utils import Trainer
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


class GAT_Trainer(Trainer):
    def __init__(self, graph, features, labels, train_mask, test_mask, val_mask, model, epochs, path):
        super(GAT_Trainer, self).__init__(graph, features, labels, train_mask,
                                          test_mask, val_mask, model, epochs, path)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        max_score = 0
        for epoch in range(self.epochs):

            logits = self.model(self.features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'epoch - {epoch} | Loss - {loss}')

            score = self.evaluate()
            if score>max_score:
                max_score = score
                torch.save(self.model.state_dict(), self.path) # saves model with the best validation accuracy

    def evaluate(self, val=True):
        logits = self.model(self.features)
        logp = F.log_softmax(logits, 1)
        if val:
            inf_label = logp[self.val_mask].argmax(dim=1)
            true_label = self.labels[self.val_mask]
        else:
            inf_label = logp[self.test_mask].argmax(dim=1)
            true_label = self.labels[self.test_mask]

        return (accuracy_score(true_label.numpy(), inf_label.numpy()))





