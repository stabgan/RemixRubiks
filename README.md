# RemixRubiks

A Rubik's Cube solver using Reinforcement Learning with Deep Q-Network (DQN) implementation.

## Overview

This project implements a complete Rubik's Cube environment for reinforcement learning research. It features a 3D cube representation, all standard cube rotations, and a DQN agent that learns to solve the cube through trial and error.

## Methodology

The system uses a **Deep Q-Network (DQN)** approach with experience replay to train an agent to solve Rubik's cubes:

- **State Representation**: 54-element list representing all cube faces (6 faces × 9 squares each)
- **Action Space**: 12 possible moves (6 face rotations × 2 directions each)
- **Reward System**: Positive rewards for solving, negative for loops/invalid states
- **Experience Replay**: Stores and samples past experiences for stable learning

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| 🐍 **Core** | Python 3.6+ |
| 🧠 **ML Framework** | NumPy, Pandas |
| 🎮 **RL Environment** | Custom Gym-compatible interface |
| 📊 **Visualization** | ASCII art cube display |
| 🔄 **Training** | Deep Q-Network with experience replay |

## Installation & Usage

### Prerequisites
```bash
pip install numpy pandas
```

### Basic Cube Operations
```bash
# Run the basic cube environment
python Rubikmovement.py
```

### RL Training Environment
```bash
# Use the advanced RL environment (requires ML framework)
python Rubikmovement_newBeta.py
```

## File Structure

- **`Rubikmovement.py`** - Core cube mechanics and basic operations
- **`Rubikmovement_newBeta.py`** - Complete RL environment with DQN agent
- **`README.md`** - This documentation

## Features

✅ **Complete Cube Simulation**: All 12 standard Rubik's cube rotations  
✅ **Visual Display**: ASCII representation of cube state  
✅ **RL Environment**: OpenAI Gym-compatible interface  
✅ **DQN Agent**: Deep Q-Network with experience replay  
✅ **Configurable Training**: Adjustable hyperparameters and rewards  
✅ **Statistics Tracking**: Episode performance and convergence metrics  

## Cube Representation

The cube state is represented as a 54-element list where:
- Elements 0-8: White face (top)
- Elements 9-20: Blue/Red/Green faces (middle band)
- Elements 21-32: Orange faces (middle band continued)
- Elements 33-44: Remaining middle positions
- Elements 45-53: Yellow face (bottom)

Each face's center (5th element) remains fixed as it represents the face color.

## ⚠️ Known Issues

- **Training Convergence**: DQN may require extensive training time for complex cube states
- **Memory Usage**: Experience replay can consume significant memory for large buffer sizes
- **Reward Engineering**: Current reward system may need tuning for optimal learning

## Contributing

This is a research project exploring RL applications to combinatorial puzzles. Feel free to experiment with different reward structures, network architectures, or training algorithms.

---

*Originally created by kaustabh (2018) - Modernized and debugged*