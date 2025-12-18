# Lightweight Adaptive Tutor (LAT) System

## Overview

The **Lightweight Adaptive Tutor (LAT)** is a reinforcement learning-based intelligent tutoring system designed to optimize student learning outcomes through adaptive question difficulty selection. This project implements a Q-Learning agent that dynamically adjusts problem difficulty based on student knowledge and performance, compared against rule-based baseline strategies.

### Key Features

- **Reinforcement Learning Agent**: Implements tabular Q-Learning with ϵ-greedy exploration
- **Stochastic Student Simulator**: Models realistic student knowledge evolution and performance
- **Adaptive Difficulty Management**: Learns optimal difficulty selection through interaction
- **Comparative Analysis**: Benchmarks RL agent against two baseline tutoring strategies
- **Comprehensive Evaluation**: Statistical testing, visualization, and policy interpretability analysis

---

## Project Architecture

### Core Components

#### 1. **`environment.py`** - Student Knowledge Simulator
The MDP environment that models student-tutor interactions.

**Key Features:**
- **State Space**: Two-dimensional state $(K_t, C_t)$
  - $K_t$: Knowledge level (10 discrete levels from 0.0 to 1.0)
  - $C_t$: Consecutive response counter (-3 to +3, indicating consecutive correct/incorrect responses)
  - Total states: 70 possible states

- **Action Space**: Three difficulty levels
  - "Easy" (difficulty=1)
  - "Medium" (difficulty=2)
  - "Hard" (difficulty=3)

- **Student Response Model**: Logistic function for probability of correct answer
  $$P(\text{Correct}|S_t, a_t) = \sigma(\alpha(K_t - \beta_d))$$
  where $\alpha$ is the slope parameter and $\beta_d$ is the difficulty threshold

- **Knowledge Dynamics** (Equation 3):
  - Successful response: $K_{t+1} = K_t + \lambda \cdot d$ (learning gain proportional to difficulty)
  - Failed response: $K_{t+1} = K_t - \phi$ (penalty for incorrect answer)

- **Reward Structure**:
  - Positive rewards for correct answers (scaled by difficulty)
  - Negative rewards for incorrect answers (penalized based on difficulty)

#### 2. **`q_learning_agent.py`** - Q-Learning Agent
Implements tabular Q-Learning for optimal policy discovery.

**Features:**
- Q-table initialization: zeros $(70 \times 3)$
- **Q-Learning Update Rule**:
  $$Q(s,a) \leftarrow Q(s,a) + \eta[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$
  - Learning rate ($\eta$): 0.1
  - Discount factor ($\gamma$): 0.9

- **ϵ-Greedy Exploration**:
  - Initial $\epsilon$: 1.0
  - Decay: 0.99995 per episode
  - Minimum $\epsilon$: 0.1

- **State-Action Management**: Efficient indexing between 2D states and 1D table indices
- **Training History**: Records episode rewards and exploration rates for analysis

#### 3. **`baselines.py`** - Baseline Tutoring Strategies
Two rule-based baseline methods for comparison.

**Fixed Schedule (FS) Baseline:**
- Deterministic difficulty progression
- Cycles through Easy → Medium → Hard every 7 steps
- Simple, non-adaptive approach

**Performance Threshold (PT) Baseline:**
- Threshold-based adaptation
- Rules:
  - Increase difficulty after 2 consecutive correct responses
  - Decrease difficulty after 2 consecutive incorrect responses
- Represents a simple adaptive strategy

#### 4. **`train_evaluation.py`** - Training Orchestrator
Manages training, evaluation, and comparative analysis.

**Functions:**
- `train_agent()`: Train the Q-Learning agent (~10,000 episodes × 20 steps = 200,000 interactions)
- `evaluate_agent()`: Test trained agent on held-out episodes
- `evaluate_baseline()`: Evaluate baseline strategies
- `run_comparative_evaluation()`: Compare all three methods
- `save_results()`: Persist evaluation metrics

**Metrics Collected:**
- Knowledge gains over episodes
- Success rates (% correct responses)
- Optimal challenge rates (% in productive struggle zone: 0.3 < P(Correct) < 0.7)
- Total cumulative rewards

#### 5. **`analyze.py`** - Analysis & Visualization
Comprehensive statistical analysis and visual reporting.

**Capabilities:**
- **Training Progress Plots**: Reward trends and ϵ-decay visualization
- **Comparative Bar Charts**: Side-by-side comparison of all metrics
- **Distribution Plots**: Histogram analysis of metric distributions
- **Statistical Testing**: Paired t-tests with effect size (Cohen's d)
- **Policy Interpretability**: Heatmap visualization of learned decision policy
- **Comprehensive Reports**: Summary statistics and conclusions

#### 6. **`main.py`** - Execution Pipeline
Orchestrates the complete workflow from training to analysis.

**Pipeline Steps:**
1. Initialize training orchestrator
2. Train Q-Learning agent (10,000 episodes)
3. Save trained Q-table and training history
4. Run comparative evaluation (1,000 test episodes per method)
5. Save and load results
6. Generate visualizations
7. Perform statistical tests
8. Analyze learned policy
9. Generate final report

---

## Installation & Setup

### Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`

### Quick Start

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the Complete Pipeline**:
```bash
python main.py
```

3. **Output Files** (generated in `output/` directory):
   - `q_table.npy` - Trained Q-table
   - `training_history.pkl` - Training progress metrics
   - `results.pkl` - Evaluation results
   - `statistical_results.csv` - Statistical test results
   - `training_progress.png` - Training visualizations
   - `comparative_metrics.png` - Performance comparison charts
   - `*_distributions.png` - Metric distribution plots
   - `policy_visualization.png` - Learned policy heatmap

---

## Key Algorithms & Equations

### 1. Student Response Probability
$$P(\text{Correct}|S_t, a_t) = \frac{1}{1 + e^{-\alpha(K_t - \beta_d)}}$$

### 2. Knowledge Update
$$K_{t+1} = \begin{cases} 
K_t + \lambda_{\text{gain}} \cdot d & \text{if correct} \\
K_t - \phi & \text{if incorrect}
\end{cases}$$

### 3. Q-Learning Bellman Update
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \eta \left[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]$$

### 4. Policy Extraction
$$\pi^*(s) = \arg\max_a Q(s, a)$$

### 5. Statistical Significance (Cohen's d)
$$d = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{(s_1^2 + s_2^2)/2}}$$

---

## Configuration Parameters

### Environment Parameters (`environment.py`)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `knowledge_levels` | 10 | Discrete knowledge level bins |
| `consecutive_bounds` | 7 | Range for consecutive counter (-3 to +3) |
| `alpha` | 5 | Sigmoid slope for response probability |
| `lambda_gain` | 0.05 | Learning gain coefficient |
| `phi` | 0.02 | Penalty for incorrect response |

### Q-Learning Hyperparameters (`q_learning_agent.py`)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` (η) | 0.1 | Q-value update step size |
| `discount_factor` (γ) | 0.9 | Future reward weight |
| `epsilon_init` | 1.0 | Initial exploration rate |
| `epsilon_min` | 0.1 | Minimum exploration rate |
| `epsilon_decay` | 0.99995 | Per-episode decay |

### Training Configuration (`main.py`)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_episodes` | 10,000 | Training episodes |
| `steps_per_episode` | 20 | Steps per training episode |
| `num_test_episodes` | 1,000 | Evaluation episodes per method |

---

## Expected Results

### Performance Metrics

The system typically demonstrates:

1. **Knowledge Gain**: Average improvement in student knowledge levels
   - Expected: RL agent shows higher average gains than baselines

2. **Success Rate**: Percentage of correctly answered questions
   - Expected: 70-85% with adaptive selection

3. **Optimal Challenge Rate**: Percentage of questions in productive struggle zone
   - Expected: RL agent maximizes this rate compared to fixed baselines

4. **Total Reward**: Cumulative reward over episodes
   - Expected: RL agent shows higher cumulative rewards after convergence

### Statistical Significance

Results typically show:
- Paired t-tests comparing RL vs. baselines (p < 0.05)
- Cohen's d effect sizes indicating magnitude of improvements
- More statistically significant results on optimal challenge rate metric

---

## File Structure

```
Lightweight Adaptive Tutor (LAT) system/
├── main.py                        # Main execution pipeline
├── environment.py                 # Student knowledge simulator (MDP environment)
├── q_learning_agent.py           # Q-Learning agent implementation
├── baselines.py                  # Baseline tutoring strategies (FS, PT)
├── train_evaluation.py           # Training orchestrator & evaluation
├── analyze.py                    # Analysis & visualization
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── __pycache__/                  # Python cache
└── output/                       # Generated outputs
    ├── q_table.npy              # Trained Q-table
    ├── training_history.pkl     # Training metrics
    ├── results.pkl              # Evaluation results
    ├── statistical_results.csv  # Statistical tests
    ├── training_progress.png    # Training plots
    ├── comparative_metrics.png  # Performance comparison
    ├── *_distributions.png      # Distribution plots
    └── policy_visualization.png # Policy heatmap
```

---

## Methodology Notes

### Why Q-Learning?
- **Simplicity**: Tabular representation suitable for small state spaces (70 states)
- **Optimality**: Guaranteed convergence to optimal policy
- **Interpretability**: Learned policy easily visualizable and explainable
- **Efficiency**: Fast training and inference without neural networks

### Why These Baselines?
- **Fixed Schedule (FS)**: Represents non-adaptive tutoring (baseline)
- **Performance Threshold (PT)**: Represents simple rule-based adaptation
- **Comparison**: Validates that RL provides improvements over practical alternatives

### Hyperparameter Justification
- **10 Knowledge Levels**: Sufficient granularity for student modeling
- **7-Step Consecutive Counter**: Captures productive struggle patterns (-3 to +3)
- **Learning Rate 0.1**: Balanced learning speed and stability
- **Epsilon Decay 0.99995**: Gradual transition from exploration to exploitation

---

## Future Enhancements

- **Function Approximation**: Neural network Q-Learning for larger state spaces
- **Transfer Learning**: Pre-train on synthetic student populations
- **Multi-Student**: Simultaneous tutoring of multiple students
- **Real Data Integration**: Validation with actual student interaction logs
- **Pedagogical Constraints**: Incorporate educational best practices
- **Exploration Strategies**: Thompson sampling or UCB instead of ϵ-greedy
- **Curriculum Learning**: Structured knowledge progression

---

## Troubleshooting

### Issue: Q-table file not found
**Solution**: Ensure `main.py` completes training successfully. Check `output/` directory exists.

### Issue: Low success rates
**Solution**: This may be expected with random initial students. Check if agent has converged by examining training progress plots.

### Issue: Statistical tests not running
**Solution**: Ensure evaluation completed with sufficient episodes (minimum 1,000 per method recommended).

### Issue: Slow execution
**Solution**: Reduce `num_episodes` for testing, or use multi-processing if available.

---

## Performance Optimization Tips

1. **Vectorization**: Already implemented with NumPy
2. **Batch Processing**: Modify `train_evaluation.py` to batch episodes
3. **Parallel Baselines**: Evaluate baselines in parallel
4. **Early Stopping**: Monitor convergence and stop if no improvement

---

## License

MIT License - See LICENSE file for details

---

## Contact & Support

For questions, issues, or suggestions:
- Open an issue in the repository
- Submit a pull request with improvements
- Contact: [your-email@example.com]

---

## Acknowledgments

- Built with NumPy, Matplotlib, SciPy, Pandas, Scikit-learn
- Inspired by adaptive learning research and ITS literature
- Q-Learning algorithm from Watkins & Dayan (1992)

---

**Last Updated**: December 2024
**Version**: 1.0
