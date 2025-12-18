import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from environment import StudentKnowledgeSimulator
from q_learning_agent import QLearningAgent
from baselines import FixedScheduleTutor, PerformanceThresholdTutor

class TrainingOrchestrator:
    """Orchestrates training and evaluation of RL agent and baselines."""
    
    def __init__(self):
        self.env = StudentKnowledgeSimulator()
        self.agent = QLearningAgent(
            state_size=self.env.get_total_states(),
            action_size=len(self.env.actions)
        )
        self.baselines = {
            'FS': FixedScheduleTutor(),
            'PT': PerformanceThresholdTutor()
        }
        
        # Data storage
        self.training_data = []
        self.testing_data = []
        
    def generate_training_dataset(self, num_episodes: int = 10000, 
                                 steps_per_episode: int = 20) -> List[Dict]:
        """
        Generate training dataset with ~200,000 interaction tuples.
        τ_t = (S_t, a_t, r_t, S_{t+1})
        """
        print(f"Generating training dataset: {num_episodes} episodes x {steps_per_episode} steps")
        
        dataset = []
        
        for episode in tqdm(range(num_episodes)):
            state = self.env.reset()
            state_idx = self.env.get_state_index()
            episode_data = []
            total_reward = 0
            
            for step in range(steps_per_episode):
                # Agent selects action
                action_idx = self.agent.get_action(state_idx, training=True)
                action = self.env.actions[action_idx]
                
                # Environment step
                next_state, reward, done, info = self.env.step(action)
                next_state_idx = self.env.get_state_index()
                
                # Store interaction tuple
                interaction = {
                    'episode': episode,
                    'step': step,
                    'state': state,
                    'state_idx': state_idx,
                    'action': action,
                    'action_idx': action_idx,
                    'reward': reward,
                    'next_state': next_state,
                    'next_state_idx': next_state_idx,
                    'done': done,
                    'info': info
                }
                
                episode_data.append(interaction)
                
                # Update agent
                self.agent.update(state_idx, action_idx, reward, next_state_idx, done)
                
                # Update for next step
                state = next_state
                state_idx = next_state_idx
                total_reward += reward
            
            # Decay epsilon after each episode
            self.agent.decay_epsilon()
            
            # Record training progress
            self.agent.save_training_step(episode, total_reward, self.agent.epsilon)
            
            # Add to dataset
            dataset.extend(episode_data)
        
        self.training_data = dataset
        print(f"Training dataset generated: {len(dataset)} interaction tuples")
        return dataset
    
    def train_agent(self, num_episodes: int = 10000, steps_per_episode: int = 20):
        """Train the Q-learning agent."""
        print(f"Training Q-learning agent for {num_episodes} episodes...")
        self.generate_training_dataset(num_episodes, steps_per_episode)
        print("Training completed!")
    
    def evaluate_agent(self, num_episodes: int = 1000, 
                      steps_per_episode: int = 20) -> Dict[str, Any]:
        """Evaluate trained agent on test dataset."""
        print(f"Evaluating agent on {num_episodes} test episodes...")
        
        metrics = {
            'knowledge_gains': [],
            'success_rates': [],
            'optimal_challenge_rates': [],
            'total_rewards': [],
            'episode_data': []
        }
        
        for episode in tqdm(range(num_episodes)):
            state = self.env.reset()
            state_idx = self.env.get_state_index()
            initial_knowledge = self.env.get_knowledge_value()
            
            episode_correct = 0
            episode_optimal = 0
            episode_reward = 0
            episode_steps = []
            
            for step in range(steps_per_episode):
                # Agent selects action (no exploration during evaluation)
                action_idx = self.agent.get_action(state_idx, training=False)
                action = self.env.actions[action_idx]
                
                # Environment step
                next_state, reward, done, info = self.env.step(action)
                next_state_idx = self.env.get_state_index()
                
                # Track metrics
                if info['correct']:
                    episode_correct += 1
                
                if self.env.is_optimal_challenge(info['probability_correct']):
                    episode_optimal += 1
                
                episode_reward += reward
                
                # Store step data
                step_data = {
                    'state': state,
                    'action': action,
                    'correct': info['correct'],
                    'probability_correct': info['probability_correct'],
                    'reward': reward,
                    'knowledge_gain': info['knowledge_gain']
                }
                episode_steps.append(step_data)
                
                # Update state
                state = next_state
                state_idx = next_state_idx
            
            # Calculate episode metrics
            final_knowledge = self.env.get_knowledge_value()
            knowledge_gain = final_knowledge - initial_knowledge
            
            success_rate = episode_correct / steps_per_episode * 100
            optimal_challenge_rate = episode_optimal / steps_per_episode * 100
            
            # Store metrics
            metrics['knowledge_gains'].append(knowledge_gain)
            metrics['success_rates'].append(success_rate)
            metrics['optimal_challenge_rates'].append(optimal_challenge_rate)
            metrics['total_rewards'].append(episode_reward)
            metrics['episode_data'].append(episode_steps)
        
        # Calculate averages
        metrics['avg_knowledge_gain'] = np.mean(metrics['knowledge_gains'])
        metrics['avg_success_rate'] = np.mean(metrics['success_rates'])
        metrics['avg_optimal_challenge_rate'] = np.mean(metrics['optimal_challenge_rates'])
        metrics['avg_total_reward'] = np.mean(metrics['total_rewards'])
        
        return metrics
    
    def evaluate_baseline(self, baseline_name: str, num_episodes: int = 1000,
                         steps_per_episode: int = 20) -> Dict[str, Any]:
        """Evaluate a baseline strategy."""
        print(f"Evaluating {baseline_name} baseline...")
        
        baseline = self.baselines[baseline_name]
        metrics = {
            'knowledge_gains': [],
            'success_rates': [],
            'optimal_challenge_rates': [],
            'total_rewards': [],
            'episode_data': []
        }
        
        for episode in tqdm(range(num_episodes)):
            self.env.reset()
            initial_knowledge = self.env.get_knowledge_value()
            
            episode_correct = 0
            episode_optimal = 0
            episode_reward = 0
            last_correct = None
            
            for step in range(steps_per_episode):
                state = self.env.get_state()
                
                # Baseline selects action
                if baseline_name == 'PT':
                    action = baseline.get_action(state, last_correct=last_correct)
                else:
                    action = baseline.get_action(state)
                
                # Environment step
                next_state, reward, done, info = self.env.step(action)
                
                # Track metrics
                last_correct = info['correct']
                if info['correct']:
                    episode_correct += 1
                
                if self.env.is_optimal_challenge(info['probability_correct']):
                    episode_optimal += 1
                
                episode_reward += reward
            
            # Calculate episode metrics
            final_knowledge = self.env.get_knowledge_value()
            knowledge_gain = final_knowledge - initial_knowledge
            
            success_rate = episode_correct / steps_per_episode * 100
            optimal_challenge_rate = episode_optimal / steps_per_episode * 100
            
            # Store metrics
            metrics['knowledge_gains'].append(knowledge_gain)
            metrics['success_rates'].append(success_rate)
            metrics['optimal_challenge_rates'].append(optimal_challenge_rate)
            metrics['total_rewards'].append(episode_reward)
        
        # Calculate averages
        metrics['avg_knowledge_gain'] = np.mean(metrics['knowledge_gains'])
        metrics['avg_success_rate'] = np.mean(metrics['success_rates'])
        metrics['avg_optimal_challenge_rate'] = np.mean(metrics['optimal_challenge_rates'])
        metrics['avg_total_reward'] = np.mean(metrics['total_rewards'])
        
        return metrics
    
    def run_comparative_evaluation(self, num_test_episodes: int = 1000):
        """Run comparative evaluation of all methods."""
        print("Running comparative evaluation...")
        
        # Evaluate RL agent
        rl_metrics = self.evaluate_agent(num_test_episodes)
        
        # Evaluate baselines
        fs_metrics = self.evaluate_baseline('FS', num_test_episodes)
        pt_metrics = self.evaluate_baseline('PT', num_test_episodes)
        
        results = {
            'RL': rl_metrics,
            'FS': fs_metrics,
            'PT': pt_metrics
        }
        
        # Statistical comparison
        print("\n=== Comparative Results ===")
        for metric_name in ['avg_knowledge_gain', 'avg_success_rate', 
                           'avg_optimal_challenge_rate', 'avg_total_reward']:
            print(f"\n{metric_name.replace('_', ' ').title()}:")
            print(f"  RL Agent: {results['RL'][metric_name]:.4f}")
            print(f"  Fixed Schedule: {results['FS'][metric_name]:.4f}")
            print(f"  Performance Threshold: {results['PT'][metric_name]:.4f}")
        
        return results
    
    def save_results(self, results: Dict, filename: str = 'results.pkl'):
        """Save evaluation results to file."""
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {filename}")