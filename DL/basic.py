import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import networkx as nx
import gym

from tubemap import tubemap

g = nx.Graph()
g.add_nodes_from(tubemap.keys())

for k, v in tubemap.items():
    g.add_edges_from(([(k,t) for t in v]))

print(g.edges)

#pos = nx.spring_layout(g)
#nx.draw_networkx_nodes(g, pos)
#nx.draw_networkx_edges(g, pos)
#nx.draw_networkx_labels(g, pos)
#plt.show()

    
R = np.matrix(np.zeros(shape=(60,60)))
R = pd.DataFrame(R)
for node in g.nodes:
    for x in g[node]:
        if node == 'Marylebone':
            R[x, node] = 100

Q = np.matrix(np.zeros(shape = (60,60)))
Q -= 100
Q = pd.DataFrame(Q)
for node in g.nodes:
    for x in g[node]:
        Q[node,x] = 0
        Q[x, node] = 0

print(Q.head())
