# DCA Environment

A Distributed Channel Access (DCA) environment implemented with OpenAI Gym-like interface for Reinforcement Learning (RL) and Multi-Agent RL experiments. The environment simulates multiple nodes competing for channel access using CSMA/CA protocol.

## Features

- Configurable number of nodes and time slots
- Binary Exponential Backoff (BEB) and Random backoff strategies
- Real-time visualization of channel access patterns
- Support for both single-agent and multi-agent training

## Installation

1. Install dependencies:
```bash
pip install numpy matplotlib ipykernel
```

2. Clone the repository:
```bash
git clone <repository-url>
cd dca_env
```

## Usage

### Basic Usage

```python
from DCAenv import DCAEnv

# Create environment with 5 nodes and 1000 time slots
env = DCAEnv(num_nodes=5, time_slots=1000)

# Reset the environment
state = env.reset()

# Run simulation
for _ in range(1000):
    # Get actions from your agents
    actions = [agent.act(state) for agent in your_agents]
    
    # Step the environment
    next_state, rewards, done, info = env.step(actions)
    
    # Render the environment (optional)
    env.render()
    
    if done:
        break
```

### Using with Jupyter Notebook

The environment includes example notebooks (`practice.ipynb`, `practice2.ipynb`, and `practice3.ipynb`) that demonstrate different use cases:

1. Basic environment interaction
2. Training with different backoff strategies
3. Multi-agent scenarios

To run the notebooks:
```bash
jupyter notebook
```

Then open any of the practice notebooks to see example implementations.

## Environment Parameters

- `num_nodes`: Number of nodes competing for channel access (default: 5)
- `time_slots`: Total number of time slots for simulation (default: 1000)
- `render_mode`: Visualization mode (default: None)

## Agent Configuration

The environment supports two backoff strategies:
- Binary Exponential Backoff (BEB)
- Random backoff

Configure your agents with:
```python
from agents import CSMA_CA_Agent

agent = CSMA_CA_Agent(
    node_id=0,
    cw_min=16,  # Minimum contention window
    cw_max=1024,  # Maximum contention window
    strategy='beb'  # or 'random'
)
```
