import gymnasium as gym 
from env.DCAenv import DCAenv
from agents import CSMA_CA_Agent
import matplotlib.pyplot as plt


def main(): 
    num_nodes = 5
    time_slots = 500
    env = DCAenv(num_nodes=num_nodes, time_slots=time_slots, render_mode='rgb_array')
    agents = [CSMA_CA_Agent(i, cw_min=2, cw_max=16) for i in range(num_nodes)]

    state, _ = env.reset()
    for _ in range(time_slots):
        actions = [agent.act(state) for agent in agents]
        state, reward, done, truncated, _ = env.step(actions)
        env.render()
        if done or truncated:
            break
    
    if not env.NOTEBOOK:
        plt.show()

if __name__ == "__main__":
    main()