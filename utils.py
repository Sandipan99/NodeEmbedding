from dgl.data import CoraGraphDataset
import dgl


def load_cora_data():
    dataset = CoraGraphDataset()
    g = dataset[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    val_mask = g.ndata['val_mask']
    g.set_n_initializer(dgl.init.zero_initializer)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, test_mask, val_mask


class Trainer:
    def __init__(self, graph, features, labels, train_mask, test_mask, val_mask, model, epochs, path):
        self.g = graph
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.val_mask = val_mask
        self.model = model
        self.epochs = epochs
        self.path = path

    def train(self):
        pass # to be overloaded by corresponding architecture


