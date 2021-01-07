import networkx as nx
from random import randint

def formatNetwork(fname,delim): # make sure the node ids are from 0 to |nodes| - 1
    nodes = {}
    edge_list = []
    count = 0
    with open(fname) as fs:
        for line in fs:
            try:
                u,v = tuple(map(int,line.strip().split(delim)))
                if u not in nodes:
                    nodes[u] = count
                    count+=1
                if v not in nodes:
                    nodes[v] = count
                    count+=1
                edge_list.append((nodes[u],nodes[v]))
            except:
                continue

    return edge_list

def buildGraph(fname,delim=' ',formatted=False):

    G = nx.Graph()

    edge_list = []
    nodes = []
    if not formatted:
        edge_list = formatNetwork(fname, delim)

    else:
        with open(fname) as fs:
            for line in fs:
                try:
                    u, v = line.strip().split(delim)
                    edge_list.append((int(u), int(v)))
                    if u not in nodes:
                        nodes.append(u)
                    if v not in nodes:
                        nodes.append(v)
                except:
                    continue


    for e in edge_list:
        G.add_edge(*e)

    return G


def convertNextSentence(all_walks):
    #  takes a bunch of random walks and then randomly split it to form a next
    #  sentence format ... a '\t' is inserted as delimeter to separate sentences....

    all_walks_nxt = []

    for w in all_walks:
        x = randint(2,len(w)-1)
        w_t = w[:x]+['\t']+w[x:]
        sent = ' '.join(w_t)
        all_walks_nxt.append(sent)

    return all_walks_nxt

