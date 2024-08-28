import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

SCALE_OFFSET = 1e-6

class VHPInferenceNetwork(nn.Module):
    '''
    The inference network for the  variational hierarchical distribution q(zeta | z_1). 
    This is deterministic according to the paper, indicating a dirac delta distribution.'''
    def __init__(self, continuous_dim, network):
        super(VHPInferenceNetwork, self).__init__()
        self.network = network
        
    def forward(self, z_1):
        return self.network(z_1)
    
class ContinuousTransition(nn.Module):
    '''
    The continuous transition model p(z_t | z_{t-1})
    '''
    def __init__(self, continuous_dim, num_base_matrices, mixture_network):
        super(ContinuousTransition, self).__init__()
        self.mixture_network = mixture_network
        self.base_matrices_z = nn.Parameter(torch.randn(num_base_matrices, continuous_dim, continuous_dim))
        self.base_matrices_q = nn.Parameter(torch.randn(num_base_matrices, continuous_dim, continuous_dim))
        
    def forward(self, z_t_1):
        
        # Compute the mixture weights
        mixture_weights = self.mixture_network(z_t_1) # B, T, num_base_matrices
        mixed_matrix_z = torch.einsum('btd, dmm -> btmm', mixture_weights, self.base_matrices_z)
        mean = torch.einsum('btmm, btm -> btm', mixed_matrix_z, z_t_1)
        mixed_matrix_q = torch.einsum('btd, dmm -> btmm', mixture_weights, self.base_matrices_q)
        covariance = torch.einsum('btmm, btm -> btm', mixed_matrix_q, z_t_1)
        covariance = F.softplus(covariance)

        dist = D.independent.Independent(D.normal.Normal(mean, covariance + SCALE_OFFSET), 1)

        return dist

class AuxiliaryInferenceNetwork(nn.Module):
    '''
    The inference network for the variational distribution q(a_t | x_t). 
    This is deterministic according to the paper, indicating a dirac delta distribution.
    '''
    def __init__(self, network):
        super(AuxiliaryInferenceNetwork, self).__init__()
        self.network = network
        
    def forward(self, x):
        return self.network(x)
    

class ObservationNetwork(nn.Module):
    '''
    The observation model p(x_t | a_t). This is a Gaussian distribution with diagonal covariance matrix.
    '''
    def __init__(self, obs_dim, network):
        super(ObservationNetwork, self).__init__()
        self.network = network
        self.obs_dim = obs_dim
        
    def forward(self, a):
        output = self.network(a)
        mean = output[..., :self.obs_dim]
        covariance = F.softplus(output[..., self.obs_dim:])
        assert covariance.shape[-1] == self.obs_dim
        dist = D.independent.Independent(D.normal.Normal(mean, covariance + SCALE_OFFSET), 1)
        
        return dist