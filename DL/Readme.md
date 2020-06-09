# Q learning and Double Deep Q Networks

There are several files present in this directory, for setting up of the tasks and experiments outlined in the report.

## Q learning

```basic.py``` - contains all the functions needed to setup a Q learning algorithm on a series of nodes and edges in dictionary form

```tubemap.py``` - contains all nodes, edges and some helper functions for the London Underground zone 1 map

```basic.ipynb``` - contains the graph and initial R matrix as described in the report

```experiments.ipynb``` - contains all experiments conducted in the report, as well as some extra which were excluded

## DDQN

```space_invaders.ipynb``` - contains the training and testing code for running a double deep Q network on the Atari 2600 game Space Invaders, in OpenAI's gym environment

## Usage 

The notebook files are self-explanatory. For Q learning, can conduct any desired learning by performing:

```python3 basic.py start_node end_node num_episodes policy_type learning_rate discount_rate initial_parameter```

This will return a cumulative rewards graph, and the final shortest path, with its length 
