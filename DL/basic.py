#!/usr/bin/env python3

import sys
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import networkx as nx

import tubemap
from tubemap import tubemap_dictionary

# tubemap is names of locations, but for ease of coding we want to convert to numbers

# We use networkx to simplify things down. Mostly to quickly get a list of connections:

def create_networkx_graph(tubemap_dictionary):
  dct = {}
  for i, loc in enumerate(list(tubemap_dictionary.keys())):
    dct[loc] = i
  tubemap_new = {}
  for k, v in tubemap_dictionary.items():
    tubemap_new[dct[k]] = v
  for k, v in tubemap_new.items():
    _ = []
    for place in v:
      _.append(dct[place])
    tubemap_new[k] = _

  return nx.Graph(tubemap_new)

def initialise_R(nx_graph, end_loc):
  R = np.matrix(np.zeros(shape=(59,59)))
  for node in nx_graph.nodes:
    for x in nx_graph[node]:
      if node == end_loc:
        R[x, node] = 100
  return R

def initialise_Q(nx_graph):
  Q = np.matrix(np.zeros(shape = (59, 59)))
  Q -= 100
  for node in nx_graph.nodes:
    for x in nx_graph[node]:
      Q[node, x] = 0
      Q[x, node] = 0
  return Q

def next_node(start, threshold, graph):
  rand = random.uniform(0, 1)
  if rand < threshold:
    sample = list(dict(graph[start]).keys())
  else:
    sample = np.where(Q[start,] == np.max(Q[start,]))[1]
  next_node = np.random.choice(sample)
  return next_node

def update_Q(node1, node2, learning_rate, gamma):
  max_idx = np.where(Q[node2,] == np.max(Q[node2,]))[1]
  if max_idx.shape[0] > 1:
    max_idx = int(np.random.choice(max_idx))
  else:
    max_idx = int(max_idx)
  max_val = Q[node2, max_idx]
  max_val = Q[node2, max_idx]
  Q[node1, node2] = int((1-learning_rate)*Q[node1, node2] + learning_rate*(R[node1, node2] + gamma*max_val))

def learn(threshold, learning_rate, gamma, num_episodes, graph):
  for i in range(num_episodes):
    start = np.random.randint(0, 59)
    next_n = next_node(start, threshold, graph)
    update_Q(start, next_n, learning_rate, gamma)
  return Q

def shortest_path(start, end, Q):
  path = [start]
  next_node = np.argmax(Q[start,])
  path.append(next_node)
  while next_node != end:
    next_node = np.argmax(Q[next_node,])
    path.append(next_node)
  return [tubemap.num_convert(pat) for pat in path]

if __name__ == '__main__':
  inp = input('Start station: ')
  inp2 = input('End station: ')
  if inp not in tubemap_dictionary.keys() or inp2 not in tubemap_dictionary.keys():
    print("Break yourself fool")
    e
  else:
    inp = int(tubemap.place_convert(inp)) 
    inp2 = int(tubemap.place_convert(inp2))
  g = create_networkx_graph(tubemap_dictionary)
  R = initialise_R(g, inp2)
  Q = initialise_Q(g)

  learn(0.5, 0.8, 0.8, 20000, g)
  print(shortest_path(inp, inp2, Q))

