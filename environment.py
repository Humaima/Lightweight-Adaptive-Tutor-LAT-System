import numpy as np
from typing import Tuple, Dict, Any

class StudentKnowledgeSimulator:
    """
    Stochastic simulator modeling student performance and knowledge evolution.
    Implements the MDP environment for the tutoring system.
    """
    
    def __init__(self):
        # State parameters
        self.knowledge_levels = 10  # Discrete levels from 0.0 to 1.0
        self.consecutive_bounds = 7  # -3 to +3
        
        # Action space
        self.actions = ["Easy", "Medium", "Hard"]
        self.action_to_difficulty = {"Easy": 1, "Medium": 2, "Hard": 3}
        
        # Difficulty thresholds (βd)
        self.difficulty_thresholds = {1: 0.3, 2: 0.5, 3: 0.7}
        
        # Learning parameters
        self.alpha = 5  # Slope of sigmoid function
        self.lambda_gain = 0.05  # Learning gain coefficient
        self.phi = 0.02  # Penalty for failure
        
        # State variables
        self.reset()
    
    def reset(self) -> Tuple[int, int]:
        """Reset the simulator to a random initial state."""
        # Random initial knowledge level (0-9 corresponding to 0.0-0.9)
        self.Kt = np.random.randint(0, self.knowledge_levels)
        # Reset consecutive counter to neutral (index 3 = 0)
        self.Ct = 3
        return self.get_state()
    
    def get_state(self) -> Tuple[int, int]:
        """Get current state as (knowledge_level, consecutive_counter)."""
        return (self.Kt, self.Ct)
    
    def get_state_index(self) -> int:
        """Convert state tuple to a single index for Q-table."""
        return self.Kt * self.consecutive_bounds + self.Ct
    
    def get_state_from_index(self, idx: int) -> Tuple[int, int]:
        """Convert index back to state tuple."""
        Kt = idx // self.consecutive_bounds
        Ct = idx % self.consecutive_bounds
        return (Kt, Ct)
    
    def get_total_states(self) -> int:
        """Return total number of possible states."""
        return self.knowledge_levels * self.consecutive_bounds
    
    def get_knowledge_value(self) -> float:
        """Convert discrete knowledge level to continuous value (0.0-1.0)."""
        return self.Kt / (self.knowledge_levels - 1)
    
    def probability_correct(self, action: str) -> float:
        """
        Calculate probability of correct answer using logistic function.
        P(Correct|St, at) = σ(α * (Kt - βd))
        """
        difficulty = self.action_to_difficulty[action]
        beta_d = self.difficulty_thresholds[difficulty]
        Kt_value = self.get_knowledge_value()
        
        # Sigmoid function
        z = self.alpha * (Kt_value - beta_d)
        probability = 1 / (1 + np.exp(-z))
        
        return probability
    
    def step(self, action: str) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        Returns: (next_state, reward, done, info)
        """
        # Calculate probability of correct answer
        p_correct = self.probability_correct(action)
        
        # Simulate student response
        is_correct = np.random.random() < p_correct
        difficulty = self.action_to_difficulty[action]
        
        # Update consecutive counter
        if is_correct:
            self.Ct = min(self.Ct + 1, self.consecutive_bounds - 1)
        else:
            self.Ct = max(self.Ct - 1, 0)
        
        # Update knowledge level based on equation (3)
        Kt_value = self.get_knowledge_value()
        
        if is_correct:
            # Successful learning: K(t+1) = Kt + λ * d
            delta_k = self.lambda_gain * difficulty
        else:
            # Failed attempt: K(t+1) = Kt - φ
            delta_k = -self.phi
        
        # Update knowledge value
        new_k_value = max(0.0, min(1.0, Kt_value + delta_k))
        
        # Convert back to discrete level
        self.Kt = int(round(new_k_value * (self.knowledge_levels - 1)))
        
        # Calculate reward based on difficulty and correctness
        reward = self.calculate_reward(is_correct, difficulty)
        
        # Done is always False for single steps, controlled by episodes
        done = False
        
        info = {
            "correct": is_correct,
            "probability_correct": p_correct,
            "difficulty": difficulty,
            "knowledge_gain": delta_k
        }
        
        return self.get_state(), reward, done, info
    
    def calculate_reward(self, is_correct: bool, difficulty: int) -> float:
        """
        Calculate reward based on difficulty and correctness.
        Higher utility for correct responses on difficult material.
        """
        if is_correct:
            if difficulty == 1:  # Easy
                return 1.0
            elif difficulty == 2:  # Medium
                return 2.0
            else:  # Hard
                return 3.0
        else:
            if difficulty == 1:  # Easy
                return -0.5
            elif difficulty == 2:  # Medium
                return -1.0
            else:  # Hard
                return -1.5
    
    def is_optimal_challenge(self, p_correct: float) -> bool:
        """Check if probability of correct answer indicates productive struggle."""
        return 0.3 < p_correct < 0.7