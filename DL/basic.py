#!/usr/bin/env python3

import numpy as np
import networkx as nx

import tubemap

# This converts our tubemap dictionary to ids in alphabetical order. 
# For example, the first entry Aldgate: [Liverpool Street, Tower Hill] -> 0: [33, 57]
# It then converts the dictionary into a networkx graph, because getting each connection is easier using networkx
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

# We initialise R by putting a reward of 100 wherever the action is going to the desired end location, and 0s everywhere else
def initialise_R(nx_graph, end_loc):
  R = np.matrix(np.zeros(shape=(59,59)))
  for node in nx_graph.nodes:
    for x in nx_graph[node]:
      if node == end_loc:
        R[x, node] = 100
  return R

# We initialise Q by giving every state, action pair -100, and then giving any actual connection (state, action) 0 
def initialise_Q(nx_graph):
  Q = np.matrix(np.zeros(shape = (59, 59)))
  Q -= 100
  for node in nx_graph.nodes:
    for x in nx_graph[node]:
      Q[node, x] = 0
      Q[x, node] = 0
  return Q

# We check if epsilon (which we set) is less than a random number between 0 and 1. 
# If rand < epsilon, the algorithm explores, if rand >= epsilon, we get all states and choose one with the highest reward (randomly if more than 1)
def next_node(start, epsilon, graph):
  rand = np.random.uniform()
  if rand < epsilon:
    sample = list(dict(graph[start]).keys())
  else:
    sample = np.where(Q[start,] == np.max(Q[start,]))[1]
  next_node = np.random.choice(sample)
  return next_node

# Updates the Q-matrix using the Bellman(?) equation
def update_Q(state, action, learning_rate, gamma):
  max_idx = np.where(Q[action,] == np.max(Q[action,]))[1]
  if max_idx.shape[0] > 1:
    max_idx = np.random.choice(max_idx)
  Q[state, action] = int((1-learning_rate)*Q[state, action] + learning_rate*(R[state, action] + gamma* Q[action, max_idx]))

# Starts randomly for a set amount of episodes, updating Q as it goes along
# Implemented greedy-epsilon policy, where epsilon is reduced on each episode
def learn(epsilon, learning_rate, gamma, num_episodes, graph, greedy=False):
  for i in range(num_episodes):
    start = np.random.randint(0, 59)
    next_n = next_node(start, epsilon, graph)
    if greedy == True and epsilon < 0.5:
      epsilon *= 0.9999
    elif greedy == True and epsilon >= 0.5:
      epsilon *= 0.99999
    update_Q(start, next_n, learning_rate, gamma)
  return Q

# Finds the shortest path from start by finding the highest action reward at each state until the end
# Returns a string of the stations in the shortest path separated by ->
def shortest_path(start, end, Q):
  path = [start]
  next_node = np.argmax(Q[start,])
  path.append(next_node)
  while next_node != end:
    next_node = np.argmax(Q[next_node,])
    path.append(next_node)
  return '->'.join([tubemap.num_convert(station) for station in path])

if __name__ == '__main__':
  while True:
    start = input('Start station: ').title()
    end = input('End station: ').title()
    if start not in tubemap.tubemap_dictionary.keys() or end not in tubemap.tubemap_dictionary.keys():
      print("Stations were invalid, please input again")
      continue
    else:
      start = int(tubemap.place_convert(start)) 
      end = int(tubemap.place_convert(end))
      break
  g = create_networkx_graph(tubemap.tubemap_dictionary)
  R = initialise_R(g, end)
  Q = initialise_Q(g)

  learn(epsilon = 0.5, learning_rate = 0.8, gamma = 0.8, num_episodes = 20000, graph = g, greedy=True)
  print(shortest_path(start, end, Q))
