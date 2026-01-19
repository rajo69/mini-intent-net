"""
This module defines the simulated optical network environment.

The environment is built using networkx and is designed to be partially compatible
with the Gymnasium (formerly Gym) API, allowing a DRL agent to interact with it.
"""
import networkx as nx
import numpy as np

class OpticalNetworkEnv:
    """
    A simulated optical network environment for a DRL agent.
    
    The agent's goal is to find an optimal path between two nodes.
    """
    def __init__(self, graph: nx.Graph, start_node: str, end_node: str):
        self.graph = graph
        self.start_node = start_node
        self.end_node = end_node
        self.current_node = None
        
        # Define action and observation space
        self.nodes = list(graph.nodes)
        self.action_space_size = len(self.nodes)
        # Observation is the current node's index
        self.observation_space_size = 1

        print("Network Environment Initialized.")
        print(f"  Nodes: {self.nodes}")
        print(f"  Start Node: {self.start_node}, End Node: {self.end_node}")

    def reset(self):
        """Resets the environment to the starting node."""
        self.current_node = self.start_node
        return self.nodes.index(self.current_node)

    def step(self, action_index: int):
        """
        The agent takes a step in the environment.

        Args:
            action_index: The index of the node to move to.

        Returns:
            A tuple (next_state, reward, done).
        """
        target_node = self.nodes[action_index]
        
        # Check if the move is valid (i.e., there is an edge)
        if target_node not in self.graph[self.current_node]:
            # Invalid move, penalize heavily and stay in the same state
            return self.nodes.index(self.current_node), -100, False

        # Valid move, calculate reward based on the "cost" of the path
        reward = -self.graph[self.current_node][target_node].get('weight', 1)
        
        # Now update the current node
        self.current_node = target_node
        
        done = self.current_node == self.end_node
        if done:
            # Goal reached, give a large positive reward
            reward = 100

        return self.nodes.index(self.current_node), reward, done

def create_default_graph():
    """Creates a default graph for the simulation."""
    G = nx.Graph()
    nodes = ["A", "B", "C", "D", "E", "F"]
    G.add_nodes_from(nodes)
    edges = [
        ("A", "B", {'weight': 1}),
        ("A", "C", {'weight': 2}),
        ("B", "D", {'weight': 3}),
        ("C", "D", {'weight': 1}),
        ("C", "E", {'weight': 5}),
        ("D", "F", {'weight': 2}),
        ("E", "F", {'weight': 1})
    ]
    G.add_edges_from(edges)
    print("Default network graph created.")
    return G
