import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import sys
import time
from agents import CSMA_CA_Agent

if 'ipykernel' in sys.modules:
    from IPython import display


class AccessPoint:
    def __init__(self):
        self.channel_busy = False

    def receive(self, nodes):
        if len(nodes) > 1:
            return "COLLISION"
        elif len(nodes) == 1:
            return "ACK"
        else:
            return "IDLE"

    def reset(self):
        self.channel_busy = False

class DCAenv:
    RENDER_SLOTS = 50
    NOTEBOOK = 'ipykernel' in sys.modules

    def __init__(self, num_nodes=5, time_slots=1000, render_mode=None):
        self.num_nodes = num_nodes
        self.render_mode = render_mode.lower() if render_mode else None
        self.access_point = AccessPoint()
        self.time_slots = time_slots
        self.t = 0
        self._initialize_render_data()

    def reset(self):
        self.access_point.reset()
        self.t = 0
        self._initialize_render_data()
        channel = self.access_point.receive([])
        state = [channel, False]
        return state, {}

    def step(self, actions):
        ready_nodes = [i for i, a in enumerate(actions) if a[0] == 1]
        channel = self.access_point.receive(ready_nodes)
        
        collision_occured = channel == "COLLISION"
        reward = self.compute_reward(channel, ready_nodes)
        
        # Update render data
        self.state_data = np.zeros((self.num_nodes, 1))
        if channel == 'ACK':
            self.state_data[ready_nodes[0]] = 1
        elif channel == 'COLLISION':
            for node in ready_nodes:
                self.state_data[node] = 2
        
        self.render_data['time_node'] = np.concatenate(
            (self.render_data['time_node'][:, 1:], self.state_data), axis=1
        )
        
        if self.render_mode == 'rgb_array':
            self.render_data['cumsum_success'].append(self.render_data['success'])
            self.render_data['cumsum_collision'].append(self.render_data['collision'])

        state = [channel, collision_occured]
        self.t += 1
        done = False
        truncated = self.t >= self.time_slots
        return state, reward, done, truncated, {}

    def compute_reward(self, channel, ready_nodes):
        self.render_data['time_slot'] += 1
        
        if channel == 'ACK':
            self.render_data['success'] += 1
            self.render_data['node_success'][ready_nodes[0]] += 1
            return 1
        elif channel == 'COLLISION':
            self.render_data['collision'] += 1
            for node in ready_nodes:
                self.render_data['node_collision'][node] += 1
            return -1
        return 0

    def render(self):
        if not self.render_mode:
            return
        
        time_node_data = self.render_data['time_node']
        
        if self.render_mode == "ansi":
            if self.NOTEBOOK:
                display.clear_output(wait=True)
            print(f"Time: {self.t}")
            for i, row in enumerate(time_node_data):
                print(f"Node {i+1}: {' '.join(map(str, row.astype(int)))}")
            print(f"Success: {self.render_data['success']}, Collision: {self.render_data['collision']}")
            time.sleep(0.1)

        elif self.render_mode == "rgb_array":
            plt.clf()
            cmap = colors.ListedColormap(['white', 'blue', 'red'])
            norm = colors.BoundaryNorm([0, 1, 2, 3], cmap.N)

            plt.subplot(211)
            plt.title(f"DCA (t = {self.t}, throughput: {self.render_data['success'] / self.render_data['time_slot']:.3f})")
            plt.imshow(time_node_data, cmap=cmap, norm=norm, aspect='auto')
            plt.ylabel('Node ID')

            plt.subplot(223)
            plt.plot(self.render_data['cumsum_success'], 'b-', label='Success')
            plt.plot(self.render_data['cumsum_collision'], 'r-', label='Collision')
            plt.legend()
            plt.grid(True)

            plt.subplot(224)
            plt.bar(range(self.num_nodes), self.render_data['node_success'].squeeze())
            
            plt.ylabel('Success Count')

            if self.NOTEBOOK:
                display.clear_output(wait=True)
                display.display(plt.gcf())
            else:
                plt.pause(0.001)

    def _initialize_render_data(self):
        self.render_data = {
            'time_node': np.zeros((self.num_nodes, self.RENDER_SLOTS)),
            'node_success': np.zeros((self.num_nodes, 1), dtype=int),
            'node_collision': np.zeros((self.num_nodes, 1), dtype=int),
            'time_slot': 0,
            'success': 0,
            'collision': 0,
            'cumsum_success': [],
            'cumsum_collision': []
        }

if __name__ == "__main__":
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