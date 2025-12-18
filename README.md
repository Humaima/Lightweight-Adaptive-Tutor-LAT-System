# Lightweight Adaptive Tutor (LAT) System

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reinforcement Learning](https://img.shields.io/badge/RL-Q--Learning-orange.svg)](https://en.wikipedia.org/wiki/Q-learning)

<img width="627" height="206" alt="image" src="https://github.com/user-attachments/assets/047fe7e7-f2f6-4682-8f97-ee4008fdadb6" />

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
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the complete pipeline
```bash
python main.py
```
4. Expected Output
After running, check the output/ directory for:
- Trained Q-table (q_table.npy)
- Training history and metrics (training_history.pkl)
- Performance comparison visualizations (comparative_metrics.png)
- Statistical analysis results (statistical_results.csv)
- Policy heatmap visualizations (policy_visualization.png)

