import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torch_kalman.standard import KalmanFilter

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
        dist = D.independent.Independent(D.normal.Normal(mean, covariance), 1)
        return dist


class InitialStateNetwork(nn.Module):
    """
    The generative model for the initial state z_1. Samples zeta from a Gaussian distribution, then transforms it with a neural network.
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

        mean = z_1[..., : self.continuous_dim]  # B, D
        covariance = torch.diag_embed(F.softplus(z_1[..., self.continuous_dim :])) + SCALE_OFFSET  # B, D, D
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
            torch.randn(num_base_matrices, continuous_dim, control_dim)
        )

        self.base_matrices_q = nn.Parameter(
            torch.randn(num_base_matrices, continuous_dim)
        )

    def forward(self, z_t, u_t):
        # Compute the mixture weights
        mixture_weights = self.mixture_network(torch.cat((z_t, u_t), dim=-1))  # B, num_base_matrices
        z_weights = torch.einsum(
            "bn, nij -> bij", mixture_weights, self.base_matrices_F
        )
        u_weights = torch.einsum(
            "bn, nij -> bij", mixture_weights, self.base_matrices_B
        )
        mixed_matrix_q = torch.einsum(
            "bn, nm -> bm", mixture_weights, self.base_matrices_q
        )
        covariance = torch.diag_embed(F.softplus(mixed_matrix_q)) + SCALE_OFFSET

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
        a_dist = D.independent.Independent(D.normal.Normal(mean, covariance), 1)
        return a_dist

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


class KFContinuousTransition(KalmanFilter):
    def __init__(
        self,
        continuous_dim: int,
        control_dim: int,
        num_base_matrices: int,
        mixture_network: nn.Module,
        compile_mode: bool = False,
    ):
        super().__init__(compile_mode=False)
        self.continuous_transition = ContinuousTransition(
            continuous_dim=continuous_dim,
            control_dim=control_dim,
            num_base_matrices=num_base_matrices,
            mixture_network=mixture_network,
        )
        if compile_mode:
            self.compiled_filter = torch.compile(self.filter)
            self.compiled_smooth = torch.compile(self.smooth)

    def _filter(
        self,
        observations,
        controls,
        observation_matrices,
        observation_covariance,
        observation_offsets,
        initial_state_mean,
        initial_state_covariance,
    ):
        """Modified filter method to handle dynamic transition parameters."""
        batch_size, n_timesteps, n_dim_obs = observations.shape
        n_dim_state = initial_state_mean.shape[-1]

        # Initialize storage tensors
        filtered_means = torch.zeros(
            batch_size, n_timesteps, n_dim_state, device=observations.device
        )
        filtered_covs = torch.zeros(
            batch_size,
            n_timesteps,
            n_dim_state,
            n_dim_state,
            device=observations.device,
        )
        predicted_means = torch.zeros_like(filtered_means)
        predicted_covs = torch.zeros_like(filtered_covs)

        # Initialize first timestep with initial state
        time_indices = torch.arange(n_timesteps, device=observations.device)
        predicted_means = torch.where(
            (time_indices == 0).view(1, -1, 1),
            initial_state_mean.unsqueeze(1),
            predicted_means,
        )
        predicted_covs = torch.where(
            (time_indices == 0).view(1, -1, 1, 1),
            initial_state_covariance.unsqueeze(1),
            predicted_covs,
        )

        # Store transition parameters for potential smoothing
        transition_matrices = []
        control_matrices = []
        transition_covariances = []

        for t in range(n_timesteps):
            # 1. Correct step (update with observation)
            _, filtered_means_t, filtered_covs_t = self._filter_correct(
                observation_matrices[:, t],
                observation_covariance[:, t],
                observation_offsets[:, t],
                predicted_means[:, t],
                predicted_covs[:, t],
                observations[:, t],
            )

            # Store filtered state
            filtered_means = torch.where(
                (time_indices == t).view(1, -1, 1),
                filtered_means_t.unsqueeze(1),
                filtered_means,
            )
            filtered_covs = torch.where(
                (time_indices == t).view(1, -1, 1, 1),
                filtered_covs_t.unsqueeze(1),
                filtered_covs,
            )

            # 2. Sample from filtered distribution (if needed)
            # Note: During inference, you might want to use the mean instead of sampling
            if self.training:
                dist = torch.distributions.MultivariateNormal(
                    filtered_means_t, filtered_covs_t
                )
                z_t = dist.rsample()
            else:
                z_t = filtered_means_t

            # 3. Predict next state (if not last timestep)
            if t < n_timesteps - 1:
                # Calculate transition parameters using current state
                F_t, B_t, Q_t = self.continuous_transition(z_t, controls[:, t])
                
                # Store transition parameters for smoothing
                transition_matrices.append(F_t)
                transition_covariances.append(Q_t)
                control_matrices.append(B_t)
                # Prepare control input with correct shape
                control_t = controls[:, t].unsqueeze(-1)  # Add dimension for matrix multiplication

                # Predict next state
                pred_mean, pred_cov = self._filter_predict(
                    F_t,  # transition matrix
                    Q_t,  # transition covariance
                    torch.zeros_like(filtered_means_t),  # transition offset
                    filtered_means_t,
                    filtered_covs_t,
                    control_t,  # now has correct shape
                    B_t,  # control matrix
                )

                # Store predicted state
                predicted_means = torch.where(
                    (time_indices == t + 1).view(1, -1, 1),
                    pred_mean.unsqueeze(1),
                    predicted_means,
                )
                predicted_covs = torch.where(
                    (time_indices == t + 1).view(1, -1, 1, 1),
                    pred_cov.unsqueeze(1),
                    predicted_covs,
                )

        # Stack transition parameters
        transition_matrices = torch.stack(transition_matrices, dim=1)
        transition_covariances = torch.stack(transition_covariances, dim=1)

        return (
            filtered_means,
            filtered_covs,
            predicted_means,
            predicted_covs,
            transition_matrices,
            transition_covariances,
            control_matrices,
        )

    def filter(
        self,
        observations,
        controls,
        observation_matrices,
        observation_covariance,
        observation_offsets,
        initial_state_mean,
        initial_state_covariance,
    ):
        return self._filter(
            observations,
            controls,
            observation_matrices,
            observation_covariance,
            observation_offsets,
            initial_state_mean,
            initial_state_covariance,
        )

    def forward(
        self,
        observations,
        controls,
        observation_matrices,
        observation_covariance,
        observation_offsets,
        initial_state_mean,
        initial_state_covariance,
        mode="filter",
    ):
        if self.compile_mode:
            (
                filtered_means,
                filtered_covs,
                predicted_means,
                predicted_covs,
                transition_matrices,
                transition_covariances,
                control_matrices,
            ) = self.compiled_filter(
                observations,
                controls,
                observation_matrices,
                observation_covariance,
                observation_offsets,
                initial_state_mean,
                initial_state_covariance,
                mode,
            )
            if mode == "filter":
                return filtered_means, filtered_covs, transition_matrices, transition_covariances, control_matrices
            elif mode == "smooth":
                smoothed_means, smoothed_covs = self.compiled_smooth(
                    filtered_means,
                    filtered_covs,
                    predicted_means,
                    predicted_covs,
                    transition_matrices,
                )
                return filtered_means, filtered_covs, transition_matrices, transition_covariances, smoothed_means, smoothed_covs, control_matrices
            else:
                raise ValueError(f"Invalid mode: {mode}")
        else:
            # Run filter
            (
                filtered_means,
                filtered_covs,
                predicted_means,
                predicted_covs,
                transition_matrices,
                transition_covariances,
                control_matrices,
            ) = self.filter(
                observations=observations,
                controls=controls,
                observation_matrices=observation_matrices,
                observation_covariance=observation_covariance,
                observation_offsets=observation_offsets,
                initial_state_mean=initial_state_mean,
                initial_state_covariance=initial_state_covariance,
            )
            if mode == "filter":
                return filtered_means, filtered_covs, transition_matrices, transition_covariances, control_matrices
            elif mode == "smooth":
                # Run smoother with stored transition parameters
                smoothed_means, smoothed_covs = self._smooth(
                    filtered_means,
                    filtered_covs,
                    predicted_means,
                    predicted_covs,
                    transition_matrices,
                )
                return filtered_means, filtered_covs, transition_matrices, transition_covariances, smoothed_means, smoothed_covs, control_matrices
            else:
                raise ValueError(f"Invalid mode: {mode}")
