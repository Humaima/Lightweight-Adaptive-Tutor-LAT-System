import numpy as np
from typing import Tuple, List
import random

class QLearningAgent:
    """
    Tabular Q-Learning agent for the adaptive tutoring system.
    Implements ϵ-greedy exploration with decaying epsilon.
    """
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-table initialized to zeros
        self.q_table = np.zeros((state_size, action_size))
        
        # Learning parameters
        self.learning_rate = 0.1  # η
        self.discount_factor = 0.9  # γ
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99995  # Decay per episode
        
        # Tracking
        self.training_history = []
    
    def get_action(self, state_idx: int, training: bool = True) -> int:
        """
        Select action using ϵ-greedy policy.
        Returns: action index
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.action_size)
        else:
            # Exploit: action with highest Q-value
            return np.argmax(self.q_table[state_idx])
    
    def update(self, state_idx: int, action_idx: int, 
               reward: float, next_state_idx: int, done: bool) -> None:
        """
        Update Q-table using Q-learning update rule.
        Q(s,a) ← Q(s,a) + η * [r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state_idx, action_idx]
        
        if done:
            target_q = reward
        else:
            max_future_q = np.max(self.q_table[next_state_idx])
            target_q = reward + self.discount_factor * max_future_q
        
        # Update Q-value
        self.q_table[state_idx, action_idx] = current_q + \
            self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self) -> None:
        """Linearly decay epsilon for exploration-exploitation balance."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_training_step(self, episode: int, total_reward: float, 
                          epsilon: float) -> None:
        """Record training progress."""
        self.training_history.append({
            'episode': episode,
            'total_reward': total_reward,
            'epsilon': epsilon
        })
    
    def get_policy(self) -> np.ndarray:
        """Extract greedy policy from Q-table."""
        return np.argmax(self.q_table, axis=1)
    
    def save_q_table(self, filename: str) -> None:
        """Save Q-table to file."""
        np.save(filename, self.q_table)
    
    def load_q_table(self, filename: str) -> None:
        """Load Q-table from file."""
        self.q_table = np.load(filename)