import numpy as np
from typing import Tuple, List

class BaselineTutor:
    """Base class for rule-based tutoring strategies."""
    
    def __init__(self):
        self.actions = ["Easy", "Medium", "Hard"]
        self.action_to_idx = {"Easy": 0, "Medium": 1, "Hard": 2}
        self.idx_to_action = {0: "Easy", 1: "Medium", 2: "Hard"}
    
    def get_action(self, state: Tuple[int, int], **kwargs) -> str:
        """Get action based on current state and strategy."""
        raise NotImplementedError
    
    def get_action_idx(self, state: Tuple[int, int], **kwargs) -> int:
        """Get action index."""
        action = self.get_action(state, **kwargs)
        return self.action_to_idx[action]


class FixedScheduleTutor(BaselineTutor):
    """
    Fixed Schedule (FS) baseline.
    Increases difficulty regardless of student performance.
    """
    
    def __init__(self):
        super().__init__()
        self.step_count = 0
    
    def get_action(self, state: Tuple[int, int], **kwargs) -> str:
        """Increase difficulty every 7 steps, cycling through difficulties."""
        Kt, Ct = state
        
        # Simple fixed schedule: cycle through difficulties
        difficulty_level = (self.step_count // 7) % 3
        self.step_count += 1
        
        if difficulty_level == 0:
            return "Easy"
        elif difficulty_level == 1:
            return "Medium"
        else:
            return "Hard"


class PerformanceThresholdTutor(BaselineTutor):
    """
    Performance-Threshold (PT) baseline.
    Adjusts difficulty based on consecutive correct/incorrect responses.
    """
    
    def __init__(self):
        super().__init__()
        self.current_difficulty_idx = 0  # Start with Easy
        self.consecutive_correct = 0
        self.consecutive_incorrect = 0
    
    def get_action(self, state: Tuple[int, int], **kwargs) -> str:
        """Adjust difficulty based on performance history."""
        Kt, Ct = state
        last_correct = kwargs.get('last_correct', None)
        
        if last_correct is not None:
            if last_correct:
                self.consecutive_correct += 1
                self.consecutive_incorrect = 0
            else:
                self.consecutive_incorrect += 1
                self.consecutive_correct = 0
            
            # Rule: Increase difficulty after 2 consecutive correct
            if self.consecutive_correct >= 2:
                self.current_difficulty_idx = min(self.current_difficulty_idx + 1, 2)
                self.consecutive_correct = 0
            
            # Rule: Decrease difficulty after 2 consecutive incorrect
            elif self.consecutive_incorrect >= 2:
                self.current_difficulty_idx = max(self.current_difficulty_idx - 1, 0)
                self.consecutive_incorrect = 0
        
        return self.idx_to_action[self.current_difficulty_idx]