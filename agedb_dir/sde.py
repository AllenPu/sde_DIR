import torch
import torch.nn as nn

class SDE(torch.nn.Module):
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self):
        super().__init__(batch_size, state_size=512, brownian_size=1)
        #
        self.batch_size, self.state_size, self.brownian_size = batch_size, state_size, brownian_size
        #
        self.mu = nn.Linear(self.state_size, 
                                self.state_size)
        self.sigma = nn.Linear(self.state_size, 
                                self.state_size * self.brownian_size)

    # Drift
    def f(self, t, y):
        return self.mu(y)  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        return self.sigma(y).view(self.batch_size, 
                                  self.state_size, 
                                  self.brownian_size)