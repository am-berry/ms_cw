#!/usr/bin/env python3 

import numpy as np
import networkx as nx

class Q(nx.Graph):
    def __init__(self, graph):
        super(Q, self).__init__() 
        self.sz = len(graph) 
        Q = np.matrix(np.zeros(shape = (sz,sz))) - 100
        for node in graph.nodes:
