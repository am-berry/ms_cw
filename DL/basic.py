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
  R = np.matrix(np.zeros(shape=(len(nx_graph),len(nx_graph))))
  for node in nx_graph.nodes:
    for x in nx_graph[node]:
      if node == end_loc:
        R[x, node] = 100
  return R

# We initialise Q by giving every state, action pair -100, and then giving any actual connection (state, action) 0 
def initialise_Q(nx_graph):
  Q = np.matrix(np.zeros(shape = (len(nx_graph), len(nx_graph))))
  Q -= 100
  for node in nx_graph.nodes:
    for x in nx_graph[node]:
      Q[node, x] = 0
      Q[x, node] = 0
  return Q

# We check if epsilon (which we set) is less than a random number between 0 and 1. 
# If rand < epsilon, the algorithm explores, if rand >= epsilon, we get all states and choose one with the highest reward (randomly if more than 1)
def epsilon_policy(state, epsilon, graph, Q):
  rand = np.random.uniform()
  if rand < epsilon:
    sample = list(dict(graph[state]).keys())
  else:
    sample = np.where(Q[state,] == np.max(Q[state,]))[1]
  action = np.random.choice(sample)
  return action

# Boltzmann (softmax) policy - calculates the probabilities of each action for a state, then selects pseurandomly based on these probabilities
def boltzmann_policy(state, tau, graph, Q):
  exp_values  = np.exp(Q[state,] / tau)
  probs = exp_values / np.sum(exp_values)
  action = np.random.choice(range(Q.shape[0]), p = probs.tolist()[0])
  return action

# Updates the Q-matrix using the Bellman equation
def update_Q(state, action, learning_rate, gamma, Q, R):
  og = Q.sum()
  max_idx = np.where(Q[action,] == np.max(Q[action,]))[1]
  if max_idx.shape[0] > 1:
    max_idx = np.random.choice(max_idx)
  Q[state, action] = int((1-learning_rate)*Q[state, action] + learning_rate*(R[state, action] + gamma* Q[action, max_idx]))
  return Q.sum() - og

# Starts randomly for a set amount of episodes, updating Q as it goes along
# Implemented greedy-epsilon policy, where epsilon is reduced on each episode
def learn(Q, R, learning_rate, gamma, num_episodes, graph, policy, parameter):
  assert policy == 'boltzmann' or policy == 'epsilon', 'please input either \'boltzmann\' or \'epsilon\' as policy'
  cumulative = [0]*num_episodes
  for i in range(num_episodes):
    start = np.random.randint(0, len(graph))
    if policy == 'boltzmann':
      next_action = boltzmann_policy(start, parameter, graph, Q)
      if i % 500 == 0:
        parameter *= 0.99
    elif policy == 'epsilon':
      next_action = epsilon_policy(start, parameter, graph, Q)
      if parameter < 0.5:
        parameter *= 0.9999
      else:
        parameter *= 0.99999
    cumulative[i] = update_Q(start, next_action, learning_rate, gamma, Q, R)
  return Q

# Finds the shortest path from start by finding the highest action reward at each state until the end
# Returns a string of the stations in the shortest path separated by ->
def shortest_path(start, end, Q):
  path = [start]
  next_action = np.argmax(Q[start,])
  path.append(next_action)
  while next_action != end:
    next_action = np.argmax(Q[next_action,])
    path.append(next_action)
  return '->'.join([tubemap.num_convert(station) for station in path])

if __name__ == '__main__':
  while True:
    start = input('Start station: ').strip().title()
    end = input('End station: ').strip().title()
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

  learn(Q, R, learning_rate = 0.8, gamma = 0.8, num_episodes = 20000, graph = g, policy = 'epsilon', parameter = 0.8)
  print(shortest_path(start, end, Q))
