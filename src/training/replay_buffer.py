import numpy as np
import random
from agent import LearnSample


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with FIFO circular eviction and IS weights."""
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = 0.6   # prioritization exponent
        self.pos = 0       # circular buffer write position

    def add(self, sample: LearnSample, td_error: float):
        priority = (abs(td_error) + 1e-5) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
            self.priorities.append(priority)
        else:
            # Circular FIFO eviction — preserves recency over stale high-priority samples
            self.buffer[self.pos] = sample
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> tuple[list[LearnSample], list[int], list[float]]:
        n = len(self.buffer)
        priorities_arr = np.array(self.priorities[:n], dtype=np.float32)
        total = float(np.sum(priorities_arr))
        probs = priorities_arr / max(1e-8, total)
        indices = np.random.choice(n, size=batch_size, p=probs, replace=True)
        samples = [self.buffer[i] for i in indices]

        # Importance sampling weights to correct for priority sampling bias
        weights = [(1.0 / (n * probs[i])) ** beta for i in indices]
        max_w = max(weights)
        weights = [w / max_w for w in weights]  # Normalize by max for stability

        return samples, indices, weights

    def update_priorities(self, indices: list[int], errors: list[float]):
        for idx, err in zip(indices, errors):
            if idx < len(self.priorities):
                self.priorities[idx] = (abs(err) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)
