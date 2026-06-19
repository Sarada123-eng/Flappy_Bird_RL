from collections import deque
import random

class ExperienceReplay:
    def __init__(self, max_len, seed=None):
        self.memory = deque([], maxlen=max_len)
        if seed is not None:
            random.seed(seed)
    
    def append(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)