"""
This module will contain the DRL agent.

For this project, we will use a simple Q-learning algorithm as a proof-of-concept
to demonstrate the agent's ability to learn a policy in the environment.
"""
import numpy as np

class QLearningAgent:
    """A simple Q-learning agent."""
    def __init__(self, state_size: int, action_size: int, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay_rate=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay_rate
        self.epsilon_min = 0.01

        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))
        print("Q-Learning Agent Initialized.")

    def choose_action(self, state: int):
        """
        Chooses an action using an epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            # Explore: choose a random action
            return np.random.randint(self.action_size)
        else:
            # Exploit: choose the best action from Q-table
            return np.argmax(self.q_table[state, :])

    def learn(self, state: int, action: int, reward: float, next_state: int):
        """
        Updates the Q-table using the Bellman equation.
        """
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])
        
        # Q-learning formula
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

    def update_exploration_rate(self):
        """Decays the exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

