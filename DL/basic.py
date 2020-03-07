#!/usr/bin/env python3

import random

import numpy as np
import networkx as nx

import tubemap
from tubemap import tubemap_dictionary

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

def update_Q(state, action, learning_rate, gamma):
  max_idx = np.where(Q[action,] == np.max(Q[action,]))[1]
  if max_idx.shape[0] > 1:
    max_idx = int(np.random.choice(max_idx))
  else:
    max_idx = int(max_idx)
  Q[state, action] = int((1-learning_rate)*Q[state, action] + learning_rate*(R[state, action] + gamma* Q[action, max_idx])
)

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
  return '->'.join([tubemap.num_convert(pat) for pat in path])

if __name__ == '__main__':
  while True:
    start = input('Start station: ').title()
    end = input('End station: ').title()
    if start not in tubemap_dictionary.keys() or end not in tubemap_dictionary.keys():
      print("TRY AGAIN")
      continue
    else:
      start = int(tubemap.place_convert(start)) 
      end = int(tubemap.place_convert(end))
      break
  g = create_networkx_graph(tubemap_dictionary)
  R = initialise_R(g, end)
  Q = initialise_Q(g)

  learn(0.5, 0.8, 0.8, 20000, g)
  print(shortest_path(start, end, Q))

