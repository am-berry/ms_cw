#!/usr/bin/env python3

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
  R = -1. * np.matrix(np.ones(shape=(len(nx_graph),len(nx_graph))))
  for node in nx_graph.nodes:
    for x in nx_graph[node]:
      R[x, node] = 0.
      R[node, x] = 0.
  for node in nx_graph.nodes:
    for x in nx_graph[node]:
      if x == end_loc:
        R[node,x] = 100.
        R[x,x] = 100.
  return R

def initialise_Q(nx_graph):
  return np.matrix(np.zeros(shape = (len(nx_graph), len(nx_graph))))

# We check if epsilon (which we set) is less than a random number between 0 and 1. 
# If rand < epsilon, the algorithm explores, if rand >= epsilon, we get all states and choose one with the highest reward (randomly if more than 1)
def epsilon_policy(state, epsilon, graph, Q):
  rand = np.random.uniform()
  if rand < epsilon:
    sample = list(dict(graph[state]).keys())
  else:
    sample = np.where(Q[state,] == np.max(Q[state,]))[1]
  return np.random.choice(sample)

# Boltzmann (softmax) policy - calculates the probabilities of each action for a state, then selects pseurandomly based on these probabilities
# We need to not allow the policy to choose invalid actions, else we get nonsense paths like Aldgate East -> Paddington -> Bayswater
# We do so by passing a list of valid actions, zeroing every probability not in the valid actions then renormalising said probabilities 
def boltzmann_policy(state, tau, graph, Q):
  exp_values  = np.exp(Q[state,] / tau)
  s = np.sum(exp_values)
  probs = exp_values / s 
  valid_actions = [tubemap.place_convert(x) for x in tubemap.tubemap_dictionary[tubemap.num_convert(state)]]
  probs = probs.tolist()[0]
  for i in range(len(tubemap.tubemap_dictionary)):
    if i not in valid_actions:
      probs[i] = 0
  probs = np.array(probs) / sum(probs)
  action = np.random.choice(range(Q.shape[0]), p = probs.tolist())
  return action

# Updates the Q-matrix using the Bellman equation
def update_Q(state, action, learning_rate, gamma, Q, R):
  max_idx = np.where(Q[action,] == np.max(Q[action,]))[1]
  if max_idx.shape[0] > 1:
    max_idx = np.random.choice(max_idx)
  max_idx = int(max_idx)
  Q[state, action] = (1-learning_rate)*Q[state, action] + learning_rate*(R[state, action] + gamma* Q[action, max_idx])
  if np.max(Q) > 0:
    return np.sum(Q)/np.max(Q)*100
  else:
    return 0

# Starts randomly for a set amount of episodes, updating Q as it goes along
# Parameter is updated at every step of each episode
def learn(R, learning_rate, gamma, num_episodes, graph, policy, parameter, min_parameter, start, end):
  assert policy == 'boltzmann' or policy == 'epsilon'
  Q = initialise_Q(graph)
  scores = [0]*num_episodes
  steps = [0]*num_episodes
  for i in range(num_episodes):
    cnt = 0 
    loc = np.random.randint(0, len(graph))
    while loc != end:
      cnt += 1
      start_loc = loc
      if policy == 'boltzmann':
        loc = boltzmann_policy(loc, parameter, graph, Q)
      elif policy == 'epsilon':
        loc = epsilon_policy(loc, parameter, graph, Q)
      score = update_Q(start_loc, loc, learning_rate, gamma, Q, R)
      scores[i] += score
      steps[i] += cnt
      if parameter > 0.5:
        parameter *= 0.99999
      if 0.5 > parameter > min_parameter:
        parameter *= 0.9999
  return scores, steps, Q


# Finds the shortest path from start by finding the highest action reward at each state until the end
def shortest_path(start, end, Q, graph):
  path = [start]
  next_action = start
  while next_action != end:
    next_action = np.where(Q[next_action,] == np.max(Q[next_action,]))[1]
    if next_action.shape[0] > 1:
      next_action = np.random.choice(next_action)
    next_action = int(next_action)
    path.append(next_action)
  return [tubemap.num_convert(station) for station in path]

if __name__ == '__main__':
#  while True:
#    start = input('Start station: ').strip().title()
#    end = input('End station: ').strip().title()
#    if start not in tubemap.tubemap_dictionary.keys() or end not in tubemap.tubemap_dictionary.keys():
#      print("Stations were invalid, please input again")
#      continue
#    else:
#      start = int(tubemap.place_convert(start)) 
#      end = int(tubemap.place_convert(end))
#      break
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('start', type = int)
  parser.add_argument('end', type = int)
  parser.add_argument('episodes', type = int)
  parser.add_argument('policy', type = str)
  parser.add_argument('lr', type = int)
  parser.add_argument('gamma', type = int)
  parser.add_argument('parameter', type = int)
  args = parser.parse_args()
  g = create_networkx_graph(tubemap.tubemap_dictionary)
  R = initialise_R(g, args.end)
  scores, steps, Q = learn(R, learning_rate = args.lr, gamma = args.gamma, num_episodes = args.episodes, graph = g, policy = args.policy, parameter = args.parameter, min_parameter = .05, start = args.start, end = args.end) 
  print(Q)
  plt.plot(scores)
  plt.draw()
  plt.waitforbuttonpress(0)
  plt.close()
  steps = shortest_path(args.start, args.end, Q, g)
  print(steps)
  print(len(steps))
