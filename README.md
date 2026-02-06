# NeuroSync-Manipulation

## Project Overview
This project proposes a hierarchical robot control architecture that integrates high-level reasoning with robust low-level execution. The system consists of three main components:
1. **Brain (VLM)**: Visual and semantic reasoning for goal setting.
2. **Nerve (Diffusion Policy)**: High-dimensional trajectory generation.
3. **Body (CfC-RL)**: Continuous-time neural control for robust execution and bottom-up feedback.# NeuroSync-Manipulation

## System Architecture
- **Simulator**: MuJoCo
- **Robot**: Franka Emika Panda (7-DoF)

## Installation
```bash
pip install -r requirements.txt
```

## Project Structure
- `core/`: Neural network architectures (VLM, Diffusion, CfC).

- `envs/`: MuJoCo environment wrappers and reward functions.

- `scripts/`: Training and evaluation pilines.

- `utils/`: Feedback loop logic and logging utilities.

## Usage
1. Data Collection: `python scripts/collect_data.py`

2. Pre-training (BC):`python scripts/pretrain_bc.py`

3. RL Fine-tuning: `python scripts/train_rl.py`

4. Evaluation: `python scripts/evaluate.py`
