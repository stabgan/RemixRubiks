# RemixRubiks

A 3×3 Rubik's Cube simulator and reinforcement-learning environment written in Python.

## What It Does

The project models a Rubik's Cube as a flat 54-element list (one entry per face-sticker) and provides clockwise / anti-clockwise rotation primitives for all six faces. An ASCII "unfolded cross" display lets you visualise the cube state in the terminal.

A second file (`Rubikmovement_newBeta.py`) wraps the cube in an RL environment with:

- An `RCube` class with OOP move methods
- Experience replay memory for DQN training
- Episode statistics tracking
- An OpenAI Gym / Gymnasium-compatible adapter
- A `DeepQNetworkAgent` skeleton ready for a Keras model

## Cube Representation

Each cell in the list `r` represents a single face-sticker. Colour key:

| Prefix | Colour |
|--------|--------|
| `w`    | White  |
| `b`    | Blue   |
| `r`    | Red    |
| `g`    | Green  |
| `o`    | Orange |
| `y`    | Yellow |

Every colour's 5th element (e.g. `w5`, `b5`) is the immovable centre sticker.

## 🛠 Tech Stack

| | Technology | Purpose |
|---|---|---|
| 🐍 | Python 3 | Core language |
| 🔢 | NumPy | State arrays and batch operations |
| 🐼 | Pandas | Episode statistics export |
| 🤖 | Keras / TensorFlow | DQN model (user-supplied) |

## Getting Started

```bash
# Clone
git clone https://github.com/stabgan/RemixRubiks.git
cd RemixRubiks

# Install dependencies
pip install numpy pandas

# Run the basic simulator
python Rubikmovement.py
```

### Dependencies

- Python ≥ 3.8
- `numpy`
- `pandas`
- `tensorflow` / `keras` (only needed for the DQN agent in the beta file)

## Files

| File | Description |
|------|-------------|
| `Rubikmovement.py` | Standalone cube simulator with functional-style move functions |
| `Rubikmovement_newBeta.py` | OOP cube + full RL environment, DQN agent, Gym adapter |

## ⚠️ Known Issues

- The RL environment is a work-in-progress — the DQN agent requires a user-supplied Keras model and a JSON config file to run.
- Loop / repeated-state detection in the environment is stubbed out (returns `False`).
- The Gym adapter's observation space is a placeholder `(10, 10)` array; a real integration would need a proper encoding of the 54-sticker state.

## License

MIT — see [LICENSE.md](LICENSE.md).
