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
        self.backoff_timer = 0
        self.action = [0]  # Initialize action
        self.set_new_backoff()

    def set_new_backoff(self):
        if self.strategy.lower() == 'random':
            self.backoff_timer = random.randint(1, self.cw_max)
        elif self.strategy.lower() == 'beb':
            self.backoff_timer = random.randint(1, self.current_cw)

    def reset_backoff(self, collision_occured):
        if self.strategy.lower() == 'beb':
            if collision_occured:
                self.current_cw = min(self.current_cw * 2, self.cw_max)
            else:
                self.current_cw = self.cw_min
        self.set_new_backoff()

    def decrement_backoff(self):
        if self.backoff_timer > 0:  # Changed from >=0 to >0
            self.backoff_timer -= 1

    def ready(self):
        return self.backoff_timer == 0

    def act(self, state):
        channel_state, collision_occured = state
        
        # Update backoff based on previous action and channel state
        if self.action[0] == 1:  # If node tried to transmit last time
            self.reset_backoff(collision_occured=collision_occured)
        else:
            self.decrement_backoff()

        # Set new action
        self.action = [1] if self.ready() else [0]
        return self.action

    def __repr__(self):
        return f"{self.node_id}: {'READY' if self.ready() else 'WAITING'} ({self.backoff_timer}/{self.current_cw})"