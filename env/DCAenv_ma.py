import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import random 
import sys 
import time 

if 'ipykernel' in sys.modules:
    from IPython import display

class AccessPoint: 
    """ 
    Handles channel access with multi-slot transmission 
    """
    SIF = 2         # time includes ACK returning slot 
    def __init__(self, ):
        self.channel_busy = False 
        self.current_transmitter = None   # ID of the node of current transmission 
        self.remaining_slots = 0          # Number of remaining slots for current transmission
        self.sif = self.SIF
        self.sif_mode = False 



    def receive(self, nodes, packet_length): 
        """ 
        Process transmission attempts and update channel state 

        Args: 
            nodes (list): List of node IDs attempting to transmit 
            packet_length (int): Length of the packet to be transmitted 
        
        Returns: 
            str: Channel state ('ACK', 'COLLISION', 'IDLE', or 'OCCUPIED')
        """

        if self.channel_busy:
            print(f"Channel busy, remaining slots: {self.remaining_slots}, SIF: {self.sif}")
            if len(nodes) > 0: 
                return "COLLISION"              # Interference during ongoing transmission.
            if self.remaining_slots >= 0 and self.sif > 0:              # Channel is busy but there are remaining slots for the ongoing transmission.
                if self.remaining_slots == 0: 
                    self.sif_mode = True 
                return "OCCUPIED"
            elif self.remaining_slots == 0 and self.sif == 0:
                print("fjfjfjfjfjfjfjfjfjfjf")
                self.sif_mode = False 
                self.channel_busy = False
                print("SIF=",self.sif, "SIF_MODE=",self.sif_mode)
                return "ACK" 
            # else: 
            #     return "ACK" 
        else:
            print(f"Channel idle")
            if len(nodes) > 1:
                return "COLLISION"              # Multiple nodes transmit simultaneously 
            elif len(nodes) == 1: 
                self.sif = self.SIF
                self.channel_busy = True 
                self.current_transmitter = nodes[0] 
                self.remaining_slots = packet_length
                return "OCCUPIED" 
            else:
                return "IDLE"
    
    def update(self): 
        """Decrement remaining slots and update channel state"""
        
        if self.channel_busy and self.remaining_slots > 0:
            self.remaining_slots -= 1 
        if self.remaining_slots == 0 and self.sif > 0: 
            self.sif -= 1 
        # if self.remaining_slots == 0 and self.sif == 0: 
        #     self.channel_busy = False 
            # self.current_transmitter = None 
    
    def reset(self): 
        """Reset channel """
        self.channel_busy = False 
        self.current_transmitter = None 
        self.remaining_slots = 0 
        self.sif = self.SIF



class DCAEnv(gym.Env): 
    """ 
    Decentralized POMDP for DCA, for multi-slot packet transmission and listen-before-talk (LBT) 

    Each node (agent) decides wheter to transmit(=1) or wait(=0) in each time slot if the channel is "IDLE"
    Transmission lasts for multiple time slots, and success requires no interference from other nodes 
    "ACK" issued only when the full packet has fully arrived.
    """
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 10}
    RENDER_SLOTS = 50 
    NOTEBOOK = 'ipykernel' in sys.modules 
    CHANNEL_STATE_MAPS = {"IDLE": 0, "OCCUPIED": 1, "COLLISION": 2, "ACK": 3}

    def __init__(self, 
                 num_nodes=5, 
                 max_steps=100, 
                 packet_length=3,
                 render_mode=None,):
        """ 
        Initialize the environment 

        Args: 
            num_agents (int): Number of nodes in the environment 
            max_steps (int): Maximum number of steps (time slots) the episode can last 
            packet_length (int): Number of time slots for each transmission 
            render_mode (str): The mode to render the environment (None, "human"="rgb_array", "ansi")

        """
        super().__init__() 

        self.num_nodes = num_nodes
        self.max_steps = max_steps 
        self.packet_length = packet_length
        self.render_mode = render_mode.lower() if render_mode else None 
        self.access_point = AccessPoint() 
        self.t = 0 

        self.node_ids = [f"node_{i}" for i in range(num_nodes)] 
        self.action_spaces = {node_id: gym.spaces.Discrete(2) for node_id in self.node_ids} 
        self.observation_spaces = {
            node_id: gym.spaces.Dict({"channel_state": gym.spaces.Discrete(4), 
                                       "collision": gym.spaces.Discrete(2)}) 
            for node_id in self.node_ids 
        }

        self.action_space = gym.spaces.MultiBinary(num_nodes)  # For compatibility 

        self.hidden_state = {
            "channel": "IDLE", 
            "ready_nodes": [], 
            "successful": False,    # Tracks if the current packet transmission was successful 
        }

        self._initialize_render_data() 


    def reset(self, seed=None, options=None): 
        """
        Reset the environment 
        
        Args: 
            seed (int): Random seed for reproducibility 
            options (dict): additional options 
            
        Returns: 
            observation (dict): Initial observation 
            info (dict): Additional information 
        """
        super().reset(seed=seed) 
        if seed is not None:
            np.random.seed(seed) 
        
        self.t = 0 
        # Initialize access point and hidden state 
        self.access_point.reset() 
        self.hidden_state = {"channel": "IDLE", 
                             "ready_nodes": [], 
                             "successful": False}
        self._initialize_render_data()  

        observation = self._get_observation() 
        info = {"hidden_state": self.hidden_state.copy()}

        return observation, info 

    
    def step(self, actions): 
        """Execute one step with joint actions 
        
        Args: 
            actions (dict): {agent_id: action} where action is 0 (wait) or 1 (transmit) 
            
        Returns: 
            observations (dict): Observations for each agent 
            rewards (dict): rewards per agent 
            terminated (dict): whether the episode is terminated 
            truncated (dict): whether the episode is truncated 
            info (dict): additional information (e.g., hidden_state)
        """

        action_array = np.array([actions[node_id] for node_id in self.node_ids])

        ready_nodes = np.where(action_array == 1)[0].tolist() 

        # Update channel state 
        channel = self.access_point.receive(ready_nodes, self.packet_length) 
        print(f"Left packets: {self.access_point.remaining_slots}")

        self.hidden_state["channel"] = channel 
        self.hidden_state["ready_nodes"] = ready_nodes 

        # Update the ongoing packet transmission 
        if channel == "OCCUPIED": 
            self.hidden_state["successful"] = True 
        elif channel == "COLLISION": 
            self.hidden_state["successful"] = False 
        elif channel == "ACK":
            self.hidden_state["successful"] = True 

        rewards = self._compute_rewards() 
        self.access_point.update() 

        observations = self._get_observation()  
        self.t += 1 
        terminated = {node_id: False for node_id in self.node_ids}
        truncated = {node_id: self.t >= self.max_steps for node_id in self.node_ids} 

        # update render data 
        print("CHANNEL=",channel)
        self._update_render_data() 

        info = {"hidden_state": self.hidden_state.copy()}

        return observations, rewards, terminated, truncated, info 


    def _get_observation(self): 
        """Get the current observation from the environment
        
        Returns: 
            observation (dict): Observation for each agent 
        """
        # ready_nodes = self.hidden_state["ready_nodes"]
        print(self.hidden_state)
        channel = self.hidden_state["channel"] 
        channel_state = self.CHANNEL_STATE_MAPS[channel]  
        collision = 1 if channel == "COLLISION" else 0 
        ### TODO: personalize by agent (when including D2LT)### 
        observation = {"channel_state": channel_state,                          
                       "collision": collision}
        return {node_id: observation.copy() for node_id in self.node_ids}    
    

    def _compute_rewards(self): 
        """Compute rewards based on state (observation + hidden_state)
        
        Returns: 
            rewards (dict): rewards per agent 
        """
        rewards = {node_id: 0 for node_id in self.node_ids}
        ready_nodes = self.hidden_state["ready_nodes"] 
        channel = self.hidden_state["channel"] 
        self.render_data["time_slot"] += 1 
        if channel == "ACK" and self.hidden_state["successful"]: 
            # For packet-based counts 
            self.render_data["success"] += 1 
            self.render_data["node_success"][self.access_point.current_transmitter] += 1 
            # For bit-based counts 
            self.render_data["success_packet"] += self.packet_length 
            self.render_data["node_success_packet"][self.access_point.current_transmitter] += self.packet_length 
            rewards[self.node_ids[self.access_point.current_transmitter]] = 1.0
            
        elif channel == "COLLISION": 
            self.render_data["collision"] += 1 
            for node in ready_nodes: 
                self.render_data["node_collision"][node] += 1 
                rewards[self.node_ids[node]] = -1.0 
                
        # elif channel == "BUSY": 
        #     print(f"Remaining slots: {self.access_point.remaining_slots}")
        #     if self.access_point.remaining_slots == 0 and self.hidden_state["successful"]:
        #         self.render_data["success"] += (self.packet_length - 1)
        #         self.render_data["node_success"][self.access_point.current_transmitter] += 1 
        #         rewards[self.agent_ids[self.access_point.current_transmitter]] = 1.0  
        #     elif self.access_point.remaining_slots > 0 and len(ready_nodes) > 0: 
        #         self.render_data["collision"] += 1 
        #         for node in ready_nodes: 
        #             self.render_data["node_collision"][node] += 1 
        #             rewards[self.agent_ids[node]] = -1.0  
            
        # elif channel == "BUSY" and self.access_point.remaining_slots == 0 and self.hidden_state["success"]:
        #     self.render_data["success"] += 1 
        #     self.render_data["node_success"][self.access_point.current_transmitter] += 1 
        #     rewards[self.agent_ids[self.access_point.current_transmitter]] = 1.0  
        
        return rewards 


    def render(self):
        """Render th environment in accordance to the render_mode 
        
        Returns:
            None 
        """
        if not self.render_mode: 
            return 
        
        time_node_data = self.render_data["time_node"] 

        if self.render_mode == "ansi": 
            self._render_ansi(time_node_data) 
        elif self.render_mode in ["human", "rgb_array"]:
            self._render_human(time_node_data)


    def _initialize_render_data(self): 
        """Initialize data for rendering """
        self.render_data = {
            "time_node": np.zeros((self.num_nodes, self.RENDER_SLOTS)), 
            "node_success": np.zeros((self.num_nodes, 1), dtype=int), 
            "node_collision": np.zeros((self.num_nodes, 1), dtype=int), 
            "node_success_packet": np.zeros((self.num_nodes, 1), dtype=int), 
            "time_slot": 0,
            "success": 0, 
            "success_packet": 0, 
            "collision": 0, 
            "cumsum_success": [], 
            "cumsum_success_packet": [], 
            "cumsum_collision": [],
            "throughput": [], 
            "throughput_packet": []
        }

    def _update_render_data(self): 
        """Update the render data based on the current state"""
        self.state_data = np.zeros((self.num_nodes, 1))
        ready_nodes = self.hidden_state["ready_nodes"] 
        channel = self.hidden_state["channel"] 

        if channel == "ACK": 
            self.state_data[self.access_point.current_transmitter] = 3 
        elif channel == "COLLISION": 
            for node in ready_nodes: 
                self.state_data[node] = 2  
        elif channel == "OCCUPIED": 
            # if self.access_point.remaining_slots >= 0 and self.access_point.sif > 0:
            if self.access_point.remaining_slots >= 0 and not self.access_point.sif_mode:
                self.state_data[self.access_point.current_transmitter] = 1 
            elif self.access_point.remaining_slots > 0 and len(ready_nodes) > 0: 
                for node in ready_nodes: 
                    self.state_data[node] = 2 
        self.render_data["time_node"] = np.concatenate(
            (self.render_data["time_node"][:, 1:], self.state_data), axis=1
        )

        if self.render_mode in ["human", "rgb_array"]: 
            self.render_data["cumsum_success"].append(self.render_data["success"])
            self.render_data["cumsum_success_packet"].append(self.render_data["success_packet"])
            self.render_data["cumsum_collision"].append(self.render_data["collision"])
            self.render_data["throughput"].append(self.render_data["success"] / self.t)
            self.render_data["throughput_packet"].append(self.render_data["success_packet"] / self.t)

    def _render_ansi(self, time_node_data): 
        """Render the environment in ANSI mode""" 
        if self.NOTEBOOK: 
            display.clear_output(wait=True)  
        print("*" * 35 + f"{'RENDERING DCA(t = ' + str(self.t) + ')':^40}" + "*" * 35)
        for i, row in enumerate(time_node_data):
            print(f"Node {i+1}: ", end=' ')
            for rc in row:
                CCOLOR = '\033[44m' if rc == 1 else '\033[101m' if rc == 2 else '\033[42m' if rc == 3 else '\33[7m'
                CEND = '\33[0m'
                print(f"{CCOLOR}{int(rc)}{CEND}", end=' ')
            print()
        print(f"Time Slot: {self.t}, "
                f"Success: {self.render_data['success']}, "
                f"Collision: {self.render_data['collision']}, "
                f"Throughput: {self.render_data['success_packet'] / self.t if self.t > 0 else 0:.3f}")
        print("*" * 110)
        print()
        time.sleep(0.1)
    
    def _render_human(self, time_node_data):
        plt.clf()
        cmap = colors.ListedColormap(['white', 'blue', 'red', 'lightgreen'])
        norm = colors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

        plt.subplot(311)
        plt.title(f"DCA (t = {self.t}, throughput: {self.render_data['success_packet'] / self.render_data['time_slot']:.3f})")
        plt.imshow(time_node_data, cmap=cmap, norm=norm, aspect='auto')
        plt.ylabel('Node ID')

        plt.subplot(323)
        # plt.plot(self.render_data['cumsum_success'], 'b-', label='Success')
        plt.plot(self.render_data['cumsum_success_packet'], 'b-', label='Success')
        plt.plot(self.render_data['cumsum_collision'], 'r-', label='Collision')
        plt.legend()
        plt.grid(True)

        plt.subplot(324)
        plt.bar(range(self.num_nodes), self.render_data['node_success_packet'].squeeze())
        plt.ylabel('Success Count')

        plt.subplot(313)
        # plt.plot(self.render_data["throughput"], 'b-', label='Throughput')
        plt.plot(self.render_data["throughput_packet"], 'b-', label='Throughput Packet')
        plt.legend()
        plt.grid(True)


        if self.NOTEBOOK:
            display.clear_output(wait=True)
            display.display(plt.gcf())
        else:
            plt.pause(0.001)



class CSMA_CA_Agent(): 
    """
    CSMA/CA Agent (Multi-agent style) with listen-before-talk (LBT) 
    """
    def __init__(self, 
                 agent_id, 
                 cw_min=2, 
                 cw_max=16):
        self.agent_id = agent_id 
        self.cw_min = cw_min 
        self.cw_max = cw_max 
        self.current_cw = cw_min 
        self.backoff_timer = random.randint(1, self.current_cw) 
    
    def act(self, observation): 
        """
        Decide whether to wait(0) or transmit(1) based on the observation 
        """
        channel_state = observation["channel_state"] 
        collision = observation["collision"] 

        if channel_state == 0:   # IDLE 
            if self.backoff_timer > 0: 
                self.backoff_timer -= 1 
            elif self.backoff_timer == 0: 
                return 1   # Transmit   
        else:   # BUSY, ACK, or COLLISION 
            if self.backoff_timer == 0: 
                if collision == 1: 
                    self.current_cw = min(self.cw_max, self.current_cw * 2) 
                self.backoff_timer = random.randint(1, self.current_cw) 
            elif self.backoff_timer > 0: 
                self.backoff_timer -= 1 
        return 0 



        
if __name__ == "__main__":  
    num_nodes = 5 
    max_steps = 1000 
    packet_length = 10
    render_mode = "human"
    env = DCAEnv(num_nodes=num_nodes, max_steps=max_steps, packet_length=packet_length, render_mode=render_mode) 

    legacy_agents = {f"node_{i}": CSMA_CA_Agent(i, cw_min=2, cw_max=32) for i in range(num_nodes)}

    total_reward = 0 

    obs, _ = env.reset() 
    rewards_per_node = {node_id: 0 for node_id in env.node_ids}
    for i in range(max_steps): 
        # actions = {agent_id: env.action_spaces[agent_id].sample() for agent_id in env.agent_ids}
        actions = {node_id: legacy_agents[node_id].act(obs[node_id]) for node_id in env.node_ids}
        next_obs, rewards, terminated, truncated, info = env.step(actions) 
        env.render()
        total_reward += sum(rewards.values()) 
        rewards_per_node = {node_id: rewards_per_node[node_id] + rewards[node_id] for node_id in env.node_ids}
        obs = next_obs 
    
        if any(terminated.values()) or any(truncated.values()): 
            break 
    
    if not env.NOTEBOOK: 
        plt.show()

    avg_rewards_per_node = {node_id: rewards_per_node[node_id] / max_steps for node_id in env.node_ids}
    print(f"Avg rewards: {total_reward / max_steps}")
    print(f"Avg rewards per node: {avg_rewards_per_node}")