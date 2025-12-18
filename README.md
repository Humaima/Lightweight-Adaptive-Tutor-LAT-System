# Lightweight Adaptive Tutor (LAT) System

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reinforcement Learning](https://img.shields.io/badge/RL-Q--Learning-orange.svg)](https://en.wikipedia.org/wiki/Q-learning)

<img width="627" height="206" alt="image" src="https://github.com/user-attachments/assets/da7fcc7a-475d-44c2-864c-2b980781e5f6" />

An intelligent tutoring system that uses reinforcement learning to dynamically adapt question difficulty based on student performance, optimizing learning outcomes through personalized challenge selection.

## 📋 Overview

The **Lightweight Adaptive Tutor (LAT)** implements a Q-Learning agent that selects optimal question difficulties in real-time, creating personalized learning paths. The system simulates student-tutor interactions and benchmarks the RL agent against rule-based baseline strategies to demonstrate improved learning efficiency.

### ✨ Key Features

- **🤖 Reinforcement Learning Agent**: Tabular Q-Learning with ϵ-greedy exploration
- **🎓 Stochastic Student Simulator**: Realistic knowledge evolution modeling
- **📈 Adaptive Difficulty Management**: Dynamic question selection based on performance
- **📊 Comparative Analysis**: Benchmarks against fixed schedule and threshold-based baselines
- **🔍 Comprehensive Evaluation**: Statistical testing, visualization, and policy analysis

## 🏗️ Architecture

| Component | Description |
|-----------|-------------|
| **`environment.py`** | MDP environment simulating student-tutor interactions (70 states, 3 actions) |
| **`q_learning_agent.py`** | Q-Learning agent with ϵ-greedy exploration and tabular Q-values |
| **`baselines.py`** | Two rule-based tutoring strategies for comparison |
| **`train_evaluation.py`** | Training orchestrator and evaluation pipeline |
| **`analyze.py`** | Statistical analysis and visualization tools |
| **`main.py`** | Complete execution pipeline |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/lightweight-adaptive-tutor.git
cd lightweight-adaptive-tutor
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Run the complete pipeline**
```bash
python main.py
```
## Expected Output
After running, check the output/ directory for:
- Trained Q-table (q_table.npy)
- Training history and metrics (training_history.pkl)
- Performance comparison visualizations (comparative_metrics.png)
- Statistical analysis results (statistical_results.csv)
- Policy heatmap visualizations (policy_visualization.png)

## 📊 Performance Metrics

The system evaluates tutoring strategies using:

| Metric | Description | Expected RL Performance |
|--------|-------------|-------------------------|
| 📚 **Knowledge Gain** | Improvement in student knowledge | 📈 Higher than baselines |
| ✅ **Success Rate** | % of correct answers | 🎯 70-85% with adaptation |
| ⚖️ **Optimal Challenge** | % in productive struggle zone (0.3 < P(correct) < 0.7) | 🔝 Maximized by RL |
| 🏆 **Cumulative Reward** | Total reward over episodes | 💪 Higher after convergence |

## 🧪 Methodology

### Why Q-Learning?
- **Simplicity**: Tabular representation for small state spaces (70 states)
- **Optimality**: Guaranteed convergence to optimal policy
- **Interpretability**: Easily visualizable and explainable policies
- **Efficiency**: Fast training and inference without neural networks

### Baseline Strategies
1. **Fixed Schedule (FS)**: Non-adaptive, cyclic difficulty progression
2. **Performance Threshold (PT)**: Simple rule-based adaptation based on consecutive responses

## 📈 Results & Analysis

The system generates comprehensive analysis including:
- **Training progress plots** (rewards, ϵ-decay)
- **Comparative performance bar charts**
- **Metric distribution histograms**
- **Statistical significance tests** (paired t-tests with Cohen's d)
- **Policy heatmap visualizations**

## 🔮 Future Enhancements

- [ ] **Neural network function approximation** for larger state spaces
- [ ] **Transfer learning** across student populations
- [ ] **Multi-student simultaneous tutoring**
- [ ] **Integration with real student interaction data**
- [ ] **Advanced exploration strategies** (Thompson sampling, UCB)
- [ ] **Curriculum learning** with structured knowledge progression

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Q-table file not found** | Ensure `main.py` completes training. Check `output/` directory exists. |
| **Low success rates** | May be expected with random initial students. Check training convergence. |
| **Statistical tests not running** | Ensure sufficient evaluation episodes (minimum 1,000 per method). |
| **Slow execution** | Reduce `num_episodes` for testing or implement parallel processing. |

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. **Fork the repository**
2. **Create your feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

## 📚 Acknowledgments

- Built with **NumPy, Matplotlib, SciPy, Pandas, Scikit-learn**
- Inspired by adaptive learning research and intelligent tutoring systems literature
- Q-Learning algorithm from **Watkins & Dayan (1992)**

## 📞 Contact

For questions, issues, or suggestions:
- Open an issue in the repository
- Contact: humaimaanwar123@gmail.com

---

**Last Updated**: December 2025 
**Version**: 1.0
