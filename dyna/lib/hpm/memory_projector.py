import torch


from dyna.lib.hpm.memory_field import MemoryField

class MemoryProjector:
    def __init__(self, memory: MemoryField, max_steps: int = 128, min_tau_u: float = 1e-4, min_sigma_u: float = 1e-4):
        self.memory = memory
        self.max_steps = max_steps
        self.min_tau_u = min_tau_u
        self.min_sigma_u = min_sigma_u

