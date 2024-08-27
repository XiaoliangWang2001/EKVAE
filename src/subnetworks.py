import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

class VHPInferenceNetwork(nn.Module):
    '''
    The inference network for the  variational hierarchical distribution q(zeta | z_1)'''
    def __init__(self, continuous_dim, network):
        super(VHPInferenceNetwork, self).__init__()
        self.network = network
        
    def forward(self, z_1):
        return self.network(z_1)
    
class ContinuousTransition(nn.Module):
    '''
    The continuous transition model p(z_t | z_{t-1})
    '''
    def __init__(self, continuous_dim, network):
        super(ContinuousTransition, self).__init__()
        self.network = network
        
    def forward(self, z_t_1):
        return self.network(z_t_1)