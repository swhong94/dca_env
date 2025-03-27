import numpy as np 
import random 



class CSMA_CA_Agent: 
    """
    Node class that is distributed in the network and transmits messages to the access point
    This class will be replaced with a MARL agent (or an agent) 

    Has its own backoff timer and strategy (Default: Binary Exponential Backoff (or Random), Else: Agent actions (Not implemented yet)) 
    """
    def __init__(self, node_id, cw_min, cw_max, strategy='beb'): 
        self.node_id = node_id 
        self.cw_min = cw_min 
        self.cw_max = cw_max 
        self.current_cw = cw_min 
        self.strategy = strategy  
        self.set_new_backoff() 

    def set_new_backoff(self): 
        if self.strategy.lower() == 'random':   # Fully random backoff strategy
            self.backoff_timer = random.randint(1, self.cw_max) 
        elif self.strategy.lower() == 'beb':    # Binaryr exponential backoff strategy
            self.backoff_timer = random.randint(1, self.current_cw) 

    def reset_backoff(self, collision_occured): 
        if self.strategy.lower() == 'beb': 
            # BEB -> double the current cw
            if collision_occured: 
                self.current_cw = min(self.current_cw * 2, self.cw_max) 
            else:
                self.current_cw = self.cw_min 

        self.set_new_backoff() 
        

    def decrement_backoff(self): 
        if self.backoff_timer >=0: 
            self.backoff_timer -= 1 

    def ready(self): 
        return self.backoff_timer == 0 
    
    def act(self, state): 
        
        if state[0] != "IDLE": 
            if self.action[0] == 1: 
                self.reset_backoff(collision_occured=state[1])
        self.decrement_backoff() 

        self.action = [1] if self.ready() else [0] 
        # if action[0] == 1:                  # If the node is ready to transmit 
        #     # print(f"Node {self.node_id} is ready to transmit, channel_state: {state[0]}")
        #     print(f"*****Node {self.node_id} is ready to transmit, channel_state: {state[0]}")
        #     self.reset_backoff(collision_occured=state[1])    # Reset the backoff timer 
        # self.decrement_backoff() 
        return self.action 
    
    def __repr__(self): 
        return f"{self.node_id}: {'READY' if self.ready() else 'WAITING'} ({self.backoff_timer}/{self.current_cw})"