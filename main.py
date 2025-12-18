import numpy as np
import pickle
import os
import traceback
from train_evaluation import TrainingOrchestrator
from analyze import Analyzer
from environment import StudentKnowledgeSimulator
from q_learning_agent import QLearningAgent

def main():
    """Main execution function."""
    try:
        print("="*60)
        print("Lightweight Adaptive Tutor (LAT) - Implementation")
        print("="*60)
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Step 1: Initialize orchestrator
        print("\n1. Initializing training orchestrator...")
        orchestrator = TrainingOrchestrator()
        
        # Step 2: Train the RL agent
        print("\n2. Training Q-learning agent...")
        orchestrator.train_agent(num_episodes=10000, steps_per_episode=20)
        
        # Step 3: Save trained agent
        print("\n3. Saving trained agent...")
        orchestrator.agent.save_q_table('output/q_table.npy')
        
        # Save training history for analysis
        with open('output/training_history.pkl', 'wb') as f:
            pickle.dump(orchestrator.agent.training_history, f)
        
        # Step 4: Run comparative evaluation
        print("\n4. Running comparative evaluation...")
        results = orchestrator.run_comparative_evaluation(num_test_episodes=1000)
        
        # Step 5: Save results
        print("\n5. Saving evaluation results...")
        orchestrator.save_results(results, 'output/results.pkl')
        
        # Step 6: Analyze results
        print("\n6. Analyzing results...")
        analyzer = Analyzer(results)
        
        # Load training history for plotting
        try:
            with open('output/training_history.pkl', 'rb') as f:
                training_history = pickle.load(f)
        except:
            print("Warning: Could not load training history.")
            training_history = []
        
        # Create visualizations
        print("\n7. Creating visualizations...")
        analyzer.plot_training_progress(training_history)
        analyzer.plot_comparative_metrics()
        
        # Use the safer distribution plotting method
        analyzer.plot_distributions()
        
        # Perform statistical tests
        print("\n8. Performing statistical analysis...")
        df_stats = analyzer.perform_statistical_tests()
        
        # Analyze policy interpretability
        print("\n9. Analyzing learned policy...")
        try:
            env = StudentKnowledgeSimulator()
            agent = QLearningAgent(env.get_total_states(), len(env.actions))
            
            if os.path.exists('output/q_table.npy'):
                agent.load_q_table('output/q_table.npy')
                policy_matrix = analyzer.analyze_policy_interpretability(agent.q_table, env)
            else:
                print("Warning: Q-table file not found.")
                policy_matrix = None
        except Exception as e:
            print(f"Error analyzing policy: {e}")
            policy_matrix = None
        
        # Generate final report
        print("\n10. Generating final report...")
        analyzer.generate_report(df_stats)
        
        # Save statistical results
        if not df_stats.empty:
            df_stats.to_csv('output/statistical_results.csv', index=False)
        
        print("\n" + "="*60)
        print("Implementation Complete!")
        print("Check 'output' folder for all generated files.")
        print("="*60)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("\nTraceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()