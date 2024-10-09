import torch
import torch.nn as nn

from pykalman.sqrt import BiermanKalmanFilter
from subnetworks import (
    InitialStateNetwork,
    ContinuousTransition,
    AuxiliaryInferenceNetwork,
    AuxiliaryObservationNetwork,
)


class EKVAE(nn.Module):
    """
    The EKVAE model without constrained optimization framework.
    """

    def __init__(
        self, continuous_dim, auxiliary_dim, hidden_dim, num_samples, num_base_matrices
    ):
        super(EKVAE, self).__init__()
        self.initial_state_network = InitialStateNetwork(continuous_dim, hidden_dim)
        self.continuous_transition = ContinuousTransition(
            continuous_dim, num_base_matrices, hidden_dim
        )
        self.auxiliary_inference_network = AuxiliaryInferenceNetwork(
            continuous_dim, hidden_dim
        )
        self.auxiliary_observation_network = AuxiliaryObservationNetwork(
            continuous_dim, auxiliary_dim, hidden_dim
        )

    def forward(self, x):
        B, T, D = x.shape
        a_mean, a_cov = self.auxiliary_inference_network(x)
        # create a normal dist with length B, T
        a_dist = D.normal.Normal(a_mean, a_cov).to_event(1)
        a = a_dist.rsample()

        q_a_x = a_dist.log_prob(a)  # log q(a_t | x_t)
        # entropy of q(a_t | x_t)
        entropy_q_a_x = -a_dist.entropy()

        z_1_mean, z_1_covariance = self.initial_state_network(
            B
        )  # Samples z_1 from a standard normal distribution
        self.kalman_smoother(a, z_1_mean, z_1_covariance)

    def kalman_smoother(self, a, z_1_mean, z_1_covariance):
        T = a.shape[1]
        mean_transition_history = []
        covariance_transition_history = []
        z_t_mean = z_1_mean
        z_t_covariance = z_1_covariance
        for t in range(T-1):
            mean_transition, covariance_transition = self.continuous_transition(z_t_mean, z_t_covariance)
            mean_transition_history.append(mean_transition)
            covariance_transition_history.append(covariance_transition)
        pass

        # B, T, A = a.shape
        # z_t_mean = z_1_mean
        # z_t_covariance = z_1_covariance

        # # Initialize lists to store the entire sequence history
        # z_mean_filtered = [z_1_mean]
        # z_covariance_filtered = [z_1_covariance]
        # mean_transition_history = []
        # covariance_transition_history = []

        # # Forward pass
        # for t in range(T):
        #     # Prediction step
        #     mean_transition, covariance_transition = self.continuous_transition(
        #         z_t_mean, z_t_covariance
        #     )
        #     z_t_pred_mean = mean_transition @ z_t_mean
        #     z_t_pred_covariance = (
        #         mean_transition @ z_t_covariance @ mean_transition.T
        #         + covariance_transition
        #     )

        #     # Update step
        #     mean_update = self.auxiliary_inference_network.H @ z_t_pred_mean
        #     S = z_t_pred_covariance[:, :A, :A] + self.auxiliary_observation_network.R
        #     K = z_t_pred_covariance[:, :A] @ torch.inverse(S)
        #     z_t_mean = z_t_pred_mean + K @ (a[:, t, :] - mean_update)
        #     z_t_covariance = z_t_pred_covariance - K @ S @ K.T

        #     # Store the current state in the history
        #     z_mean_filtered.append(z_t_mean)
        #     z_covariance_filtered.append(z_t_covariance)
        #     mean_transition_history.append(mean_transition)
        #     covariance_transition_history.append(covariance_transition)

        # # Convert lists to tensors
        # z_mean_filtered = torch.stack(z_mean_filtered, dim=1)
        # z_covariance_filtered = torch.stack(z_covariance_filtered, dim=1)

        # # Backward pass
        # z_mean_smoothed = [z_mean_filtered[:, -1, :]]
        # z_covariance_smoothed = [z_covariance_filtered[:, -1, :]]

        # for t in range(T - 2, -1, -1):
        #     mean_transition, covariance_transition = (
        #         mean_transition_history[t],
        #         covariance_transition_history[t],
        #     )
        #     J = z_covariance_filtered[:, t, :] @ mean_transition.T @ torch.inverse(
        #         z_covariance_filtered[:, t + 1, :]
        #     )
        #     z_t_mean = z_mean_filtered[:, t, :] + J @ (
        #         z_mean_smoothed[-1] - z_mean_filtered[:, t + 1, :]
        #     )
        #     z_t_covariance = (
        #         z_covariance_filtered[:, t, :]
        #         + P
        #         @ (z_covariance_smoothed[-1] - z_covariance_filtered[:, t + 1, :])
        #         @ P.T
        #     )

        #     z_mean_smoothed.append(z_t_mean)
        #     z_covariance_smoothed.append(z_t_covariance)

        # # Reverse the lists and convert to tensors
        # z_mean_smoothed = torch.stack(z_mean_smoothed[::-1], dim=1)
        # z_covariance_smoothed = torch.stack(z_covariance_smoothed[::-1], dim=1)

        # return z_mean_smoothed, z_covariance_smoothed
