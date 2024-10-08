import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

SCALE_OFFSET = 1e-6

class VHPInferenceNetwork(nn.Module):
    '''
    The inference network for the  variational hierarchical distribution q(zeta | z_1). 
    This is deterministic according to the paper, indicating a dirac delta distribution.'''
    def __init__(self, continuous_dim, hidden_dim):
        super(VHPInferenceNetwork, self).__init__()
        self.network = self._build_network(continuous_dim, hidden_dim)
        
    def forward(self, z_1):
        return self.network(z_1)
    
    def _build_network(self, continuous_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(continuous_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, continuous_dim)
        )
    
class InitialStateNetwork(nn.Module):
    '''
    The generative model for the initial state z_1. Samples zeta from a standard normal distribution, then transforms it with a neural network.
    '''
    def __init__(self, continuous_dim, hidden_dim):
        super(InitialStateNetwork, self).__init__()
        self.network = self._build_network(continuous_dim, hidden_dim)
        self.continuous_dim = continuous_dim

    def forward(self, batch_size):
        zeta = torch.randn(batch_size, 1)   
        z_1 = self.network(zeta)
        mean = z_1[:, :self.continuous_dim]
        covariance = F.softplus(z_1[:, self.continuous_dim:]) + SCALE_OFFSET
        return mean, covariance
    
    def _build_network(self, continuous_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * continuous_dim)
        )

class ContinuousTransition(nn.Module):
    '''
    The continuous transition model p(z_t | z_{t-1})
    '''
    def __init__(self, continuous_dim, num_base_matrices, mixture_network):
        super(ContinuousTransition, self).__init__()
        self.mixture_network = mixture_network
        self.base_matrices_z = nn.Parameter(torch.randn(num_base_matrices, continuous_dim, continuous_dim))
        self.base_matrices_q = nn.Parameter(torch.diag_embed(torch.randn(num_base_matrices, continuous_dim)))
        
    def forward(self, z_t_1):
        
        # Compute the mixture weights
        mixture_weights = self.mixture_network(z_t_1) # B, T, num_base_matrices
        mean_transition = torch.einsum('btd, dmm -> btmm', mixture_weights, self.base_matrices_z)
        mixed_matrix_q = torch.einsum('btd, dmm -> btmm', mixture_weights, self.base_matrices_q)
        covariance_transition = F.softplus(mixed_matrix_q) + SCALE_OFFSET

        return mean_transition, covariance_transition


class AuxiliaryObservationNetwork(nn.Module):
    '''
    The observation model p(a_t | z|t). This is a Linear Gaussian model, with fixed projection matrix H and diagonal covariance matrix R.
    '''
    def __init__(self, continuous_dim, auxiliary_dim, hidden_dim):
        super(AuxiliaryObservationNetwork, self).__init__()
        self.H = nn.Parameter(torch.eye(continuous_dim, auxiliary_dim), requires_grad=False)
        self.R = nn.Parameter(torch.eye(auxiliary_dim), requires_grad=False)

    def forward(self, z_t):
        mean = z_t @ self.H # B, T, D @ D, A -> B, T, A
        covariance = torch.diag_embed(self.R) # A, A
        # Add axis to covariance for broadcasting
        covariance = covariance[None, None, ...] # 1, 1, A, A
        return mean, covariance

class AuxiliaryInferenceNetwork(nn.Module):
    '''
    The inference network for the variational distribution q(a_t | x_t). 
    '''
    def __init__(self, obs_dim, auxiliary_dim, hidden_dim):
        super(AuxiliaryInferenceNetwork, self).__init__()
        self.auxiliary_dim = auxiliary_dim
        self.network = self._build_network(obs_dim, auxiliary_dim, hidden_dim)
        
    def forward(self, x):
        output = self.network(x)
        mean = output[..., :self.auxiliary_dim]
        covariance = F.softplus(output[..., self.auxiliary_dim:]) + SCALE_OFFSET
        assert covariance.shape[-1] == self.auxiliary_dim
        return mean, covariance
    
    def _build_network(self, obs_dim, auxiliary_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * auxiliary_dim)
        )

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