import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import sys
import time

from agents import CSMA_CA_Agent
from gymnasium import Env, spaces 


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

class DCAEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10} 
    RENDER_SLOTS = 50
    NOTEBOOK = 'ipykernel' in sys.modules
    CHANNEL_STATE_MAP = {
        "IDLE": 0,
        "ACK": 1,
        "COLLISION": 2
    }

    def __init__(self, num_nodes=5, max_steps=1000, render_mode=None):
        super().__init__() 

        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.render_mode = render_mode.lower() if render_mode else None
        self.access_point = AccessPoint()
        self.t = 0

        # Define action and observation spaces
        self.action_space = gym.spaces.MultiBinary(num_nodes)
        

        # Observation space only includes channel state and collision info
        self.observation_space = gym.spaces.Dict({
            "channel_state": gym.spaces.Discrete(3),  # 0=IDLE, 1=ACK, 2=COLLISION
            "collision": gym.spaces.Discrete(2)       # 0=False, 1=True
        })

        # Hidden state (not directly observable) 
        self.hidden_state = {
            "channel": "IDLE",      # Channel State
            "ready_nodes": [],      # Ready Nodes
            "D2LT": [],             # Number of timeslots to last successful transmission
            "actions": [],          # Agent Actions 
            "others": [],           # Indicator of other nodes transmission (0: no transmission, 1: transmission)
        }

        self._initialize_render_data()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        if seed is not None: 
            np.random.seed(seed) 
        
        self.t = 0
        self.access_point.reset()
        self.d2lt = np.zeros(self.num_nodes) 
        self.others = np.zeros(self.num_nodes) 
        self.hidden_state = {"channel": "IDLE", 
                             "ready_nodes": [], 
                             "D2LT": self.d2lt, 
                             "actions": [0] * self.num_nodes, 
                             "others": self.others}
        self._initialize_render_data()

        # Get initial state and info 
        observation = self._get_obs() 
        info = {"hidden_state": self.hidden_state.copy()} 

        return observation, info


    def step(self, actions):
        ready_nodes = np.where(actions == 1)[0].tolist()
        channel = self.access_point.receive(ready_nodes)
        self.d2lt = self.d2lt + 1   
        self.others = np.zeros(self.num_nodes)

        
        # Compute reward 
        reward = self._compute_reward(channel, ready_nodes)
        self.hidden_state = {
            "channel": channel,
            "ready_nodes": ready_nodes,
            "D2LT": self.d2lt, 
            "actions": actions,
            "others": self.others,
        } 

        # Get observation 
        observation = self._get_obs() 

        # Update time and termination conditions 
        self.t += 1
        terminated = False 
        truncated = self.t >= self.max_steps 
        info = {
            "hidden_state": self.hidden_state.copy(),
        }

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
        
        if self.render_mode == 'rgb_array' or self.render_mode == 'human':
            self.render_data['cumsum_success'].append(self.render_data['success'])
            self.render_data['cumsum_collision'].append(self.render_data['collision'])


        return observation, reward, terminated, truncated, info


    def _get_obs(self, ): 
        channel = self.hidden_state["channel"]
        channel_state = self.CHANNEL_STATE_MAP[channel]

        # D2LT (Number of time slots from last successful transmission)

        collision = 1 if channel == "COLLISION" else 0 
        return {"channel_state": channel_state, "collision": collision} 
    

    def _compute_reward(self, channel, ready_nodes):
        self.render_data['time_slot'] += 1
        self.others = np.ones(self.num_nodes)
        if channel == 'ACK':
            self.render_data['success'] += 1
            self.render_data['node_success'][ready_nodes[0]] += 1
            self.d2lt[ready_nodes[0]] = 0
            self.others[ready_nodes[0]] = 0
            return 1.0
        elif channel == 'COLLISION':
            self.render_data['collision'] += 1
            for node in ready_nodes:
                self.render_data['node_collision'][node] += 1
            return -1.0
        else: 
            self.others = np.zeros(self.num_nodes)
        return 0.0


    def render(self):
        if not self.render_mode:
            return
        
        time_node_data = self.render_data['time_node']
        
        if self.render_mode == "ansi":
            if self.NOTEBOOK:
                display.clear_output(wait=True)
            print("*" * 30 + f"{'RENDERING DCA(t = ' + str(self.t) + ')':^40}" + "*" * 30)
            for i, row in enumerate(time_node_data):
                print(f"Node {i+1} (D2LT={self.d2lt[i]:>4.1f}): ", end=' ')
                for rc in row:
                    CCOLOR = '\033[44m' if rc == 1 else '\033[101m' if rc == 2 else '\33[7m'
                    CEND = '\33[0m'
                    print(f"{CCOLOR}{int(rc)}{CEND}", end=' ')
                print()
            print(f"Time Slot: {self.t}, "
                  f"Success: {self.render_data['success']} ({self.render_data['node_success'].squeeze()}), "
                  f"Collision: {self.render_data['collision']}, "
                  f"Throughput: {self.render_data['success'] / self.t if self.t > 0 else 0:.3f}")
            time.sleep(0.1)

        elif self.render_mode == "rgb_array" or self.render_mode == "human":
            plt.clf()
            cmap = colors.ListedColormap(['white', 'blue', 'red', 'gray'])  # Gray for debugging
            norm = colors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

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







class CSMA_gym(CSMA_CA_Agent):
    """
    action returns a value instead of a singleton list
    """
    def __init__(self, agent_id, cw_min, cw_max): 
        super().__init__(agent_id, cw_min, cw_max) 

    def act(self, observation): 
        action = super().act(observation) 
        return action[0]

if __name__ == "__main__":
    num_nodes = 10
    max_steps = 1000
    env1 = DCAEnv(num_nodes=num_nodes, max_steps=max_steps, render_mode='ansi')
    # env2 = DCAEnv(num_nodes=num_nodes, max_steps=max_steps, render_mode='human')
    # env3 = DCAEnv(num_nodes=num_nodes, max_steps=max_steps, render_mode='human')
    agents = [CSMA_gym(i, cw_min=2, cw_max=16) for i in range(num_nodes)]
    # rotate_initial = np.array([1, 0, 0, 0, 0])


    observation, _ = env1.reset() 
    # observation_random, _ = env2.reset()
    actions_csma = np.array([agent.act(observation) for agent in agents])
    # action_random = env2.action_space.sample()
    # print(actions_csma, action_random)

    episode_reward_csma = 0 
    # episode_reward_random = 0 
    # episode_reward_rotate = 0 
    for _ in range(max_steps):
        # action = env.action_space.sample()
        action = np.array([agent.act(observation) for agent in agents])
        # action_random = env2.action_space.sample()
        observation, reward, terminated, truncated, info = env1.step(action) 
        # print(info)
        # observation_random, reward_random, terminated_random, truncated_random, info_random = env2.step(action_random)
        # observation_rotate, reward_rotate, terminated_rotate, truncated_rotate, info_rotate = env3.step(rotate_initial)
        env1.render()
        # env3.render()
        episode_reward_csma += reward
        # episode_reward_random += reward_random
        # episode_reward_rotate += reward_rotate

        # rotate_initial = np.roll(rotate_initial, 1)
        if terminated or truncated:
            break
    
    if not env1.NOTEBOOK:
        plt.show()

    print(f"average episode reward (CSMA): {episode_reward_csma / max_steps}")
    # print(f"average episode reward (random): {episode_reward_random / max_steps}")
    # print(f"average episode reward (rotate): {episode_reward_rotate / max_steps}")
    # # Check (verify) gym environment using checker
    # from gymnasium.utils.env_checker import check_env 
    # check_env(env)