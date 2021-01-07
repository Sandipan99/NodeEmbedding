Requirements
=============
1. pytorch
2. networkx
3. dgl


Running bert/gpt model
====================
  
  1. Input is a networkx undirected graph (G) 
  2. Function generateGraphSample will generate a subgraph of the original graph (G_s) - default sampling rate - 0.15 (fraction of edges to remove).. the obtained graph is connected and no node is removed
  3. Next step is to generate random walks ... we generate walks for both train and test .. model with the lowest error in the the test set is saved. the test set is 0.1*training size
  4. epochs = number of epochs to train the model.. it is set at 3. Also need to specify the batch size... default set at 10
  5. once the model is trained, we generate the embeddings - dictionary with node id as key and embedding tensor as the value. 4 aggregating options are available. default is set 'max pool', I have commented on the code which string to use for the corresponding aggregator.
  6. Four things are stored - embeddings (pickle file), the sampled subgraph on which the embeddings are trained and the original graph... both are stored as gpickle file.
  
Finetuning folder has the node classification and link prediction implemented...

Node classification - 
====================

1. File to run is main_nc.py
2. The function splitTrainTest splits the labeled data into train and test. The size can be controlled by the labeling rate (0.1 means only 10 percent of the data is used for training). 
3. The labels input data format is a list of tuples (node-id, label)
4. Running main_nc.py is the only file to execute and should return the test accuracy.

Link prediction -
====================

1. File to run is main_lp.py
2. The output is the average across 10 samples.

Both the tasks use a two layer feedforward network. The parameters can be set in the corresponding main files. 
