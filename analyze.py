import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from typing import Dict, Any

class Analyzer:
    """Analyzes and visualizes results from the adaptive tutoring system."""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]
    
    def plot_training_progress(self, agent_history: list):
        """Plot training progress over episodes."""
        if not agent_history:
            print("No training history available.")
            return
            
        episodes = [h['episode'] for h in agent_history]
        rewards = [h['total_reward'] for h in agent_history]
        epsilons = [h['epsilon'] for h in agent_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot rewards
        ax1.plot(episodes, rewards, 'b-', alpha=0.6)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Progress: Total Reward per Episode')
        ax1.grid(True, alpha=0.3)
        
        # Plot epsilon decay
        ax2.plot(episodes, epsilons, 'r-', alpha=0.8)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')
        ax2.set_title('Exploration Rate Decay')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_comparative_metrics(self):
        """Create comparative bar plots for all metrics."""
        methods = ['RL', 'FS', 'PT']
        metrics = ['avg_knowledge_gain', 'avg_success_rate', 
                  'avg_optimal_challenge_rate', 'avg_total_reward']
        metric_labels = ['Knowledge Gain', 'Success Rate (%)', 
                        'Optimal Challenge Rate (%)', 'Total Reward']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [self.results[method][metric] for method in methods]
            
            bars = axes[idx].bar(methods, values, color=['blue', 'orange', 'green'], alpha=0.7)
            axes[idx].set_ylabel(label)
            axes[idx].set_title(f'{label} Comparison')
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                              f'{value:.4f}', ha='center', va='bottom')
        
        plt.suptitle('Comparative Performance of Teaching Strategies', fontsize=16)
        plt.tight_layout()
        plt.savefig('output/comparative_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_distributions(self):
        """Plot distribution of key metrics."""
        metrics = ['knowledge_gains', 'success_rates', 'optimal_challenge_rates']
        metric_labels = ['Knowledge Gain', 'Success Rate (%)', 'Optimal Challenge Rate (%)']
        colors = ['blue', 'orange', 'green']
        methods = ['RL', 'FS', 'PT']
        
        # Create 3 separate plots for each metric (one for each method)
        for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            fig.suptitle(f'{label} Distribution by Method', fontsize=14)
            
            for row_idx, (method, color) in enumerate(zip(methods, colors)):
                if metric in self.results[method]:
                    data = self.results[method][metric]
                    
                    axes[row_idx].hist(data, bins=30, alpha=0.7, color=color, 
                                     edgecolor='black', density=True)
                    axes[row_idx].set_xlabel(label)
                    axes[row_idx].set_ylabel('Density')
                    axes[row_idx].set_title(f'{method} Method')
                    axes[row_idx].axvline(np.mean(data), color='red', 
                                        linestyle='--', 
                                        label=f'Mean: {np.mean(data):.3f}')
                    axes[row_idx].legend()
                    axes[row_idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'output/{metric}_distributions.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_combined_distributions(self):
        """Plot combined distributions for comparison."""
        metrics = ['knowledge_gains', 'success_rates', 'optimal_challenge_rates']
        metric_labels = ['Knowledge Gain', 'Success Rate (%)', 'Optimal Challenge Rate (%)']
        colors = {'RL': 'blue', 'FS': 'orange', 'PT': 'green'}
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            for method, color in colors.items():
                if metric in self.results[method]:
                    data = self.results[method][metric]
                    axes[idx].hist(data, bins=30, alpha=0.3, color=color, 
                                 edgecolor=color, density=True, 
                                 label=method)
            
            axes[idx].set_xlabel(label)
            axes[idx].set_ylabel('Density')
            axes[idx].set_title(f'{label} Distribution')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/combined_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def perform_statistical_tests(self):
        """Perform paired t-tests between RL agent and baselines."""
        print("=== Statistical Significance Tests ===")
        
        metrics_to_test = ['knowledge_gains', 'success_rates', 
                          'optimal_challenge_rates', 'total_rewards']
        metric_names = ['Knowledge Gain', 'Success Rate', 
                       'Optimal Challenge Rate', 'Total Reward']
        
        results_table = []
        
        for metric, metric_name in zip(metrics_to_test, metric_names):
            # Check if RL has this metric
            if metric not in self.results['RL']:
                print(f"Warning: Metric '{metric}' not found in RL results")
                continue
                
            rl_data = self.results['RL'][metric]
            
            for baseline in ['FS', 'PT']:
                # Check if baseline has this metric
                if metric not in self.results[baseline]:
                    print(f"Warning: Metric '{metric}' not found in {baseline} results")
                    continue
                    
                baseline_data = self.results[baseline][metric]
                
                # Check if we have enough data
                if len(rl_data) != len(baseline_data):
                    print(f"Warning: Data length mismatch for {metric} (RL: {len(rl_data)}, {baseline}: {len(baseline_data)})")
                    continue
                
                # Perform paired t-test
                try:
                    t_stat, p_value = stats.ttest_rel(rl_data, baseline_data)
                    
                    # Calculate effect size (Cohen's d)
                    mean_diff = np.mean(rl_data) - np.mean(baseline_data)
                    pooled_std = np.sqrt((np.std(rl_data)**2 + np.std(baseline_data)**2) / 2)
                    if pooled_std > 0:
                        cohens_d = mean_diff / pooled_std
                    else:
                        cohens_d = 0
                    
                    # Determine significance
                    significant = p_value < 0.05
                    
                    results_table.append({
                        'Metric': metric_name,
                        'Comparison': f'RL vs {baseline}',
                        'RL Mean': np.mean(rl_data),
                        f'{baseline} Mean': np.mean(baseline_data),
                        'Difference': mean_diff,
                        't-statistic': t_stat,
                        'p-value': p_value,
                        'Cohen\'s d': cohens_d,
                        'Significant (p<0.05)': 'Yes' if significant else 'No'
                    })
                    
                    print(f"\n{metric_name}: RL vs {baseline}")
                    print(f"  RL Mean: {np.mean(rl_data):.4f}, {baseline} Mean: {np.mean(baseline_data):.4f}")
                    print(f"  Difference: {mean_diff:.4f}")
                    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
                    print(f"  Cohen's d: {cohens_d:.4f}")
                    print(f"  Significant: {'Yes' if significant else 'No'}")
                    
                except Exception as e:
                    print(f"Error in statistical test for {metric_name} RL vs {baseline}: {e}")
        
        # Create pandas DataFrame for better visualization
        if results_table:
            df_results = pd.DataFrame(results_table)
            print("\n=== Summary Table ===")
            print(df_results.to_string(index=False))
            return df_results
        else:
            print("\nNo statistical tests could be performed.")
            return pd.DataFrame()
    
    def analyze_policy_interpretability(self, q_table: np.ndarray, env):
        """Analyze and interpret the learned Q-table policy."""
        print("\n=== Policy Analysis ===")
        
        if q_table is None or q_table.size == 0:
            print("No Q-table available for analysis.")
            return None
            
        # Extract greedy policy
        policy = np.argmax(q_table, axis=1)
        
        # Map indices to actions
        action_names = {0: "Easy", 1: "Medium", 2: "Hard"}
        
        # Analyze policy patterns
        print("Sample Policy Rules:")
        
        # Look at different knowledge levels
        for knowledge_level in range(0, min(env.knowledge_levels, 10), 2):  # Sample every other level
            print(f"\nKnowledge Level: {knowledge_level/(env.knowledge_levels-1):.2f}")
            
            for consecutive in range(env.consecutive_bounds):
                state_idx = knowledge_level * env.consecutive_bounds + consecutive
                if state_idx < len(policy):
                    action_idx = policy[state_idx]
                    consecutive_value = consecutive - 3  # Convert to -3 to +3 scale
                    
                    print(f"  Consecutive: {consecutive_value:2d} -> Action: {action_names.get(action_idx, 'Unknown')}")
        
        # Create policy visualization
        policy_matrix = np.zeros((env.knowledge_levels, env.consecutive_bounds))
        
        for k in range(env.knowledge_levels):
            for c in range(env.consecutive_bounds):
                state_idx = k * env.consecutive_bounds + c
                if state_idx < len(policy):
                    policy_matrix[k, c] = policy[state_idx]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(policy_matrix, cmap='viridis', aspect='auto', 
                  extent=[-3, 3, 0, 1])
        plt.colorbar(label='Action (0=Easy, 1=Medium, 2=Hard)')
        plt.xlabel('Consecutive Responses (Negative = Incorrect, Positive = Correct)')
        plt.ylabel('Knowledge Level')
        plt.title('Learned Policy Visualization')
        plt.savefig('output/policy_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return policy_matrix
    
    def generate_report(self, df_stats: pd.DataFrame):
        """Generate a comprehensive analysis report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        # Summary statistics
        print("\n1. SUMMARY STATISTICS:")
        for method in ['RL', 'FS', 'PT']:
            if method in self.results:
                print(f"\n{method}:")
                if 'avg_knowledge_gain' in self.results[method]:
                    print(f"  Avg Knowledge Gain: {self.results[method]['avg_knowledge_gain']:.4f}")
                if 'avg_success_rate' in self.results[method]:
                    print(f"  Avg Success Rate: {self.results[method]['avg_success_rate']:.2f}%")
                if 'avg_optimal_challenge_rate' in self.results[method]:
                    print(f"  Avg Optimal Challenge: {self.results[method]['avg_optimal_challenge_rate']:.2f}%")
                if 'avg_total_reward' in self.results[method]:
                    print(f"  Avg Total Reward: {self.results[method]['avg_total_reward']:.2f}")
        
        # Statistical significance
        print("\n2. STATISTICAL SIGNIFICANCE:")
        if not df_stats.empty:
            significant_comparisons = df_stats[df_stats['Significant (p<0.05)'] == 'Yes']
            if len(significant_comparisons) > 0:
                print("Significant improvements found in:")
                for _, row in significant_comparisons.iterrows():
                    print(f"  {row['Metric']}: {row['Comparison']} (p={row['p-value']:.4f}, d={row['Cohen\'s d']:.3f})")
            else:
                print("No statistically significant differences found.")
        else:
            print("No statistical analysis available.")
        
        # Overall conclusion
        print("\n3. OVERALL CONCLUSION:")
        if not df_stats.empty:
            rl_better = 0
            total_comparisons = 0
            for _, row in df_stats.iterrows():
                total_comparisons += 1
                if row['Difference'] > 0 and row['Significant (p<0.05)'] == 'Yes':
                    rl_better += 1
            
            if total_comparisons > 0:
                if rl_better >= total_comparisons * 0.75:
                    print("✓ RL agent significantly outperforms baselines on most metrics.")
                elif rl_better >= total_comparisons * 0.5:
                    print("~ RL agent shows moderate improvements over baselines.")
                elif rl_better > 0:
                    print("~ RL agent shows some improvements over baselines.")
                else:
                    print("✗ RL agent does not show significant improvement over baselines.")
            else:
                print("Insufficient data for conclusion.")
        else:
            print("Insufficient statistical data for conclusion.")