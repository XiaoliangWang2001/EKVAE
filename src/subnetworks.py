import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

SCALE_OFFSET = 1e-6


class VHPInferenceNetwork(nn.Module):
    """
    The inference network for the variational hierarchical distribution q(zeta | z_1).
    The resulting distribution is a Gaussian with diagonal covariance matrix."""

    def __init__(self, continuous_dim, hidden_dim):
        super().__init__()
        if continuous_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Dimensions must be positive")

        self.fc1 = nn.Linear(continuous_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)  # 2 outputs for mean and covariance

    def forward(self, z_1):
        x = F.relu(self.fc1(z_1))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        mean = output[..., :1]
        covariance = F.softplus(output[..., 1:]) + SCALE_OFFSET
        return mean, covariance


class InitialStateNetwork(nn.Module):
    """
    The generative model for the initial state z_1. Takes in zeta samples from a Gaussian distribution, then transforms it with a neural network.
    """

    def __init__(self, continuous_dim, hidden_dim):
        super().__init__()
        if continuous_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Dimensions must be positive")

        # Define layers as class attributes
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2 * continuous_dim)
        self.continuous_dim = continuous_dim

    def forward(self, zeta):
        # Apply layers with activations
        x = F.relu(self.fc1(zeta))
        x = F.relu(self.fc2(x))
        z_1 = self.fc3(x)

        mean = z_1[..., : self.continuous_dim]
        covariance = F.softplus(z_1[..., self.continuous_dim :]) + SCALE_OFFSET
        return mean, covariance


class ContinuousTransition(nn.Module):
    """
    The continuous transition model p(z_t | z_{t-1}, u_t)
    """

    def __init__(self, continuous_dim, control_dim, num_base_matrices, mixture_network):
        super().__init__()
        if continuous_dim <= 0 or num_base_matrices <= 0:
            raise ValueError("Dimensions must be positive")

        self.mixture_network = mixture_network
        self.base_matrices_F = nn.Parameter(
            torch.randn(num_base_matrices, continuous_dim, continuous_dim)
        )
        self.base_matrices_B = nn.Parameter(
            torch.randn(num_base_matrices, control_dim, continuous_dim)
        )

        self.base_matrices_q = nn.Parameter(
            torch.randn(num_base_matrices, continuous_dim)
        )

    def forward(self, z_t, u_t):
        # Compute the mixture weights
        mixture_weights = self.mixture_network(z_t, u_t)  # B, T, num_base_matrices
        z_weights = torch.einsum(
            "btn, nmm -> btmm", mixture_weights, self.base_matrices_F
        )
        u_weights = torch.einsum(
            "btn, nmm -> btmm", mixture_weights, self.base_matrices_B
        )
        mixed_matrix_q = torch.einsum(
            "btn, nm -> btm", mixture_weights, self.base_matrices_q
        )
        covariance = (
            torch.diag_embed(F.softplus(mixed_matrix_q)) + SCALE_OFFSET
        )

        return z_weights, u_weights, covariance


class AuxiliaryObservationNetwork(nn.Module):
    """
    The observation model p(a_t | z|t). This is a Linear Gaussian model, with fixed projection matrix H and diagonal covariance matrix R.
    """

    def __init__(self, continuous_dim, auxiliary_dim, hidden_dim):
        super(AuxiliaryObservationNetwork, self).__init__()
        self.H = nn.Parameter(
            torch.eye(continuous_dim, auxiliary_dim), requires_grad=False
        )
        self.R = nn.Parameter(torch.eye(auxiliary_dim), requires_grad=False)

    def forward(self, z_t):
        mean = z_t @ self.H  # B, T, D @ D, A -> B, T, A
        # A, A
        # Add axis to covariance for broadcasting
        covariance = self.R[None, None, ...]  # 1, 1, A, A
        return mean, covariance


class AuxiliaryInferenceNetwork(nn.Module):
    """
    The inference network for the variational distribution q(a_t | x_t).
    """

    def __init__(self, obs_dim, auxiliary_dim, hidden_dim):
        super(AuxiliaryInferenceNetwork, self).__init__()
        self.auxiliary_dim = auxiliary_dim
        self.network = self._build_network(obs_dim, auxiliary_dim, hidden_dim)

    def forward(self, x):
        output = self.network(x)
        mean = output[..., : self.auxiliary_dim]
        covariance = F.softplus(output[..., self.auxiliary_dim :]) + SCALE_OFFSET
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
            nn.Linear(hidden_dim, 2 * auxiliary_dim),
        )


class ObservationNetwork(nn.Module):
    """
    The observation model p(x_t | a_t). This is a Gaussian distribution with diagonal covariance matrix.
    """

    def __init__(self, obs_dim, network):
        super(ObservationNetwork, self).__init__()
        self.network = network
        self.obs_dim = obs_dim

    def forward(self, a):
        output = self.network(a)
        mean = output[..., : self.obs_dim]
        covariance = F.softplus(output[..., self.obs_dim :])
        assert covariance.shape[-1] == self.obs_dim
        dist = D.independent.Independent(
            D.normal.Normal(mean, covariance + SCALE_OFFSET), 1
        )

        return dist
