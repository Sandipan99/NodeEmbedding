from GAT.GraphAttentionNet import GAT, GAT_Trainer
from utils import load_cora_data


G, features, labels, train_mask, test_mask, val_mask = load_cora_data()

feature_size = features.shape[1]

hidden_dim = 8
out_dim = 7
num_heads = 2

net = GAT(G, feature_size, hidden_dim, out_dim, num_heads)

trainer = GAT_Trainer(G, features, labels, train_mask, test_mask, val_mask, net, 10, 'best_model.pt')

trainer.train()

print(trainer.evaluate(val=False))

#emb = trainer.obtainEmbeddings()