import random
import numpy as np



class ReplayBuffer:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class ReplayBuffer_I_S: # Inhibitory reward and Q-selector action
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, action_S, reward, reward_I, use_I, use_S, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, action_S, reward, reward_I, use_I, use_S, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, action_S, reward, reward_I, use_I, use_S, next_state, done = map(np.stack, zip(*batch))
        return state, action, action_S, reward, reward_I, use_I, use_S, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class ReplayBuffer_I:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, reward_I, use_I, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, reward_I, use_I,  next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, reward_I, use_I, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, reward_I, use_I, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class ReplayBuffer_I_Oct26: # original saci without use_I
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, reward_I, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, reward_I, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, reward_I, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, reward_I, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class ReplayBufferGRUs:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    # Push a trajectory of (s,a,r,s',h)
    def push(self, states, actions, rewards, next_states, dones):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = [list(states), list(actions), list(rewards), list(next_states), list(dones)]
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states = []
        final_states = []
        actions = []
        final_actions = []
        final_rewards = []
        final_next_states = []
        final_dones = []
        for i_sample, sample in enumerate(batch):
            states.append(sample[0])
            final_states.append(sample[0][-1])
            actions.append(sample[1])
            final_actions.append(sample[1][-1])
            final_rewards.append(sample[2][-1])
            final_next_states.append(sample[3][-1])
            final_dones.append(sample[4][-1])

        return final_states, final_actions, final_rewards, final_next_states, final_dones, states, actions
        
    def __len__(self):
        return len(self.buffer)