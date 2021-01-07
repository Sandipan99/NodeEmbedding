
from gensim.models import Word2Vec
from Node2Vec import node2vec
from Node2Vec.load_graph import buildGraph

def read_graph(input):
    '''
    Reads the input network in networkx.
    '''
    G = buildGraph(input)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    return G


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]

    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=workers,
                     iter=iter)
    model.wv.save_word2vec_format(output)

    return


input = 'out.ucidata-zachary'
output = 'zachary.emb'
dimensions = 128
walk_length = 80
num_walks = 10
window_size = 10
iter = 1
workers = 4
p = 1
q = 1


nx_G = read_graph(input)
G = node2vec.Graph(nx_G, p, q)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks, walk_length)
learn_embeddings(walks)
