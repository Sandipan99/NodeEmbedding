from GSage.GraphSage import GraphSAGE
from utils import load_cora_data
import torch.nn.functional as F
from GAT.GraphAttentionNet import GAT_Trainer
#import sys

#sys.path.insert(1,'../')

#from dataset_parsers import cora,citeseer,ppi_preprocessed,pubmed_diabetes


G, features, labels, train_mask, test_mask, val_mask = load_cora_data()

feature_size = features.shape[1]
n_hidden = 16
aggregator_type = 'gcn'  # mean/gcn/pool/lstm
out_dim = 7
n_layers = 1
activation = F.relu
dropout = 0.5

net = GraphSAGE(G, feature_size, n_hidden, out_dim, n_layers, activation, dropout, aggregator_type)

trainer = GAT_Trainer(G, features, labels, train_mask, test_mask, val_mask, net, 10, 'best_model.pt')

trainer.train()

print(trainer.evaluate(val=False))



#emb = trainer.obtainEmbeddings()