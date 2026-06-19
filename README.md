# Flappy Bird AI using Deep Q-Network (DQN)

## Overview

This project implements a Deep Q-Network (DQN) agent that learns to play Flappy Bird through reinforcement learning. The agent interacts with the environment, collects experiences, stores them in a replay buffer, and learns an optimal policy using neural networks.

The implementation uses:

* PyTorch
* Gymnasium
* flappy-bird-gymnasium
* Experience Replay
* Target Network
* Epsilon-Greedy Exploration

---

## Features

* Deep Q-Network (DQN) implementation from scratch
* Experience Replay Buffer
* Target Network Synchronization
* Epsilon-Greedy Exploration Strategy
* Model Checkpoint Saving
* Training and Evaluation Modes
* GPU Support (Google Colab compatible)

---

## Project Structure

```text
.
├── dqn.py                  # Neural Network Architecture
├── experience_replay.py    # Replay Buffer Implementation
├── parameters.yaml         # Hyperparameters
├── floppybirdv0.pt         # Trained Model Weights
├── runs/
│   └── floppybirdv0.log    # Training Logs
├── train.ipynb             # Training Notebook
└── README.md
```

---

## Environment

The project uses:

```python
FlappyBird-v0
```

Observation Space:

* Bird position
* Bird velocity
* Pipe information
* LiDAR-based obstacle information

Action Space:

| Action | Description |
| ------ | ----------- |
| 0      | Do Nothing  |
| 1      | Flap        |

---

## Deep Q-Network Architecture

```text
Input Layer (State Vector)
        ↓
Linear Layer (128 Neurons)
        ↓
ReLU
        ↓
Output Layer (Q-values for Actions)
```

Example:

```python
nn.Sequential(
    nn.Linear(state_dim, 128),
    nn.ReLU(),
    nn.Linear(128, action_dim)
)
```

---

## DQN Components

### Policy Network

Used to select actions and updated through gradient descent.

### Target Network

Used to generate stable target Q-values.

The target network parameters are synchronized periodically:

```python
target_dqn.load_state_dict(policy_dqn.state_dict())
```

### Experience Replay

Stores transitions:

```python
(state, action, next_state, reward, done)
```

Random minibatches are sampled during training to break temporal correlations.

---

## Bellman Equation

The target Q-value is computed using:

```math
Q_{target}
=
r + \gamma \max_a Q_{target}(s', a)
```

For terminal states:

```math
Q_{target} = r
```

---

## Hyperparameters

Example configuration:

```yaml
floppybirdv0:
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.995

  replay_buffer_size: 100000
  minibatch_size: 64

  network_sync_freq: 500

  gamma: 0.99
  alpha: 0.001

  reward_threshold: 1000
```

---

## Training

Run training:

```python
agent = FlappyBirdAgent("floppybirdv0")
agent.run(is_training=True)
```

Training process:

1. Collect experiences
2. Store in replay buffer
3. Sample minibatches
4. Update policy network
5. Periodically synchronize target network
6. Save best-performing model

---

## Testing

Run evaluation:

```python
agent = FlappyBirdAgent("floppybirdv0")
agent.run(
    is_training=False,
    render=True,
    test_episodes=10
)
```

The trained model is loaded automatically from:

```text
floppybirdv0.pt
```

---

## Results

After training for several thousand episodes:

* Agent learns to survive significantly longer than random actions
* Successfully navigates multiple pipes
* Learns stable flying behavior through reinforcement learning

---

## Future Improvements

* Double DQN (DDQN)
* Dueling DQN
* Prioritized Experience Replay
* Soft Target Network Updates
* Checkpointing Optimizer State
* TensorBoard Integration
* Hyperparameter Tuning

---

## Learning Outcomes

This project demonstrates:

* Reinforcement Learning fundamentals
* Markov Decision Processes (MDP)
* Bellman Optimality Principle
* Deep Q-Learning
* Experience Replay
* Target Networks
* PyTorch-based RL implementation

---

## Author

Sarada Prasannna Das

