from GCN.gcn import GCN_Net, GCN_Trainer
from utils import load_cora_data

#import sys

#sys.path.insert(1,'../')

#from dataset_parsers import cora,citeseer,ppi_preprocessed,pubmed_diabetes


G, features, labels, train_mask, test_mask, val_mask = load_cora_data()

feature_size = features.shape[1]

embedding_size = 16
output_size = 7

net = GCN_Net(feature_size, embedding_size, output_size)

trainer = GCN_Trainer(G, features, labels, train_mask, test_mask, val_mask, net, 10, 'best_model.pt')

trainer.train()

print(trainer.evaluate(val=False))

#emb = trainer.obtainEmbeddings()
