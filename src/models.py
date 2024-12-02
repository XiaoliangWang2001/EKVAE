import torch
import torch.nn as nn
import torch.distributions as D
from torch.distributions import kl_divergence
from ..torch_kalman import specialized_standard
from subnetworks import (
    InitialStateNetwork,
    VHPInferenceNetwork,
    ContinuousTransition,
    AuxiliaryInferenceNetwork,
    AuxiliaryObservationNetwork,
    ObservationNetwork,
    KFContinuousTransition,
)
from config import EKVAEConfig


class EKVAE(nn.Module):
    """
    The EKVAE model without constrained optimization framework.
    TODO: confirm how to handle initial state z_1
    """

    def __init__(self, config: EKVAEConfig):
        super().__init__()
        self.config = config
        self.register_buffer(
            "observation_matrix",
            torch.eye(self.config.auxiliary_dim, self.config.continuous_dim).T,
        )
        self.register_buffer(
            "observation_covariance", torch.eye(self.config.auxiliary_dim)
        )
        self.register_buffer(
            "observation_offset", torch.zeros(self.config.auxiliary_dim)
        )
        self._build_networks()

    def forward(self, x, u):
        B, T, D = x.shape
        a_dist = self.auxiliary_inference_network(x)
        a = a_dist.rsample()  # Get auxiliary variable samples, shape (B, T, D_a)
        a_inf_ll = a_dist.log_prob(a)  # shape (B, T, D_a)
        zeta = self.zeta_dist.sample(torch.Size((B, 1)))
        x_dist = self.observation_network(a)
        distortion = -x_dist.log_prob(x).sum(-1).mean()  # scalar
        initial_state_mean, initial_state_covariance = self.initial_state_network(zeta)

        (
            filtered_z_mean,
            filtered_z_cov,
            transition_matrices,
            transition_covariances,
            smoothed_z_mean,
            smoothed_z_cov,
            control_matrices,
        ) = self.kf(
            observations=a,
            controls=u,
            observation_matrices=self.observation_matrix,
            observation_covariance=self.observation_covariance,
            observation_offsets=self.observation_offset,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            mode="smooth",
        )  # shape (B, T, D_z) / (B, T, D_z, D_z) / (B, T - 1, D_z, D_z) / (B, T - 1, D_z, D_z) / (B, T, D_z) / (B, T, D_z, D_z)
        rate = self._get_rate(
            a_inf_ll,
            a,
            filtered_z_mean,
            filtered_z_cov,
            transition_matrices,
            transition_covariances,
            smoothed_z_mean,
            smoothed_z_cov,
            u,
            control_matrices,
        )
        loss = self.get_loss(distortion, rate)
        return loss

        pass

    def _build_networks(self):
        self.initial_state_network = InitialStateNetwork(
            self.config.continuous_dim, self.config.initial_state.hidden_dim
        )
        self.vhp_inference_network = VHPInferenceNetwork(
            self.config.continuous_dim, self.config.initial_state.hidden_dim
        )
        self.auxiliary_inference_network = AuxiliaryInferenceNetwork(
            self.config.continuous_dim,
            self.config.auxiliary_dim,
            self.config.auxiliary.hidden_dim,
        )
        self.auxiliary_observation_network = AuxiliaryObservationNetwork(
            self.config.continuous_dim,
            self.config.auxiliary_dim,
            self.config.auxiliary.hidden_dim,
        )
        self.num_samples = self.config.num_samples

        self.mixture_network = nn.Sequential(
            nn.Linear(
                self.config.continuous_dim + self.config.control_dim,
                self.config.kalman.mixture_net_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                self.config.kalman.mixture_net_dim, self.config.kalman.num_base_matrices
            ),
        )

        self.kf = KFContinuousTransition(
            self.config.continuous_dim,
            self.config.control_dim,
            self.config.kalman.num_base_matrices,
            self.mixture_network,
            self.config.kalman.compile_mode,
        )
        self.zeta_dist = D.normal.Normal(0, 1)

        pass

    def _get_rate(
        self,
        auxiliary_inference_log_likelihood,
        auxiliary_samples,
        filtered_z_mean,
        filtered_z_cov,
        transition_matrices,
        transition_covariances,
        smoothed_z_mean,
        smoothed_z_cov,
        control, 
        control_matrices,
    ):
        B, T, _ = auxiliary_samples.shape
        smoothed_z1_dist = D.multivariate_normal.MultivariateNormal(
            smoothed_z_mean[:, 0, :], smoothed_z_cov[:, 0, :, :]
        )
        smoothed_z1_samples = smoothed_z1_dist.rsample()  # shape (B, D_z)
        zeta_q_dist = self.vhp_inference_network(smoothed_z1_samples)
        zeta_q_samples = zeta_q_dist.rsample(torch.Size((self.num_samples,))).view(
            B * self.num_samples, 1
        )  # shape (B * self.num_samples, 1)
        kl_zeta = zeta_q_dist.log_prob(zeta_q_samples) - self.zeta_dist.log_prob(
            zeta_q_samples
        ).view(B, self.num_samples).mean(1)  # shape (B, )
        z1_given_zeta_mean, z1_given_zeta_cov = self.initial_state_network(
            zeta_q_samples
        )
        z1_given_zeta_dist = D.multivariate_normal.MultivariateNormal(
            z1_given_zeta_mean, z1_given_zeta_cov
        )  # shape (B * self.num_samples, D_z)
        kl_z1 = (
            smoothed_z1_dist.log_prob(smoothed_z1_samples)  # B, 1
            - z1_given_zeta_dist.log_prob(
                smoothed_z1_samples.unsqueeze(1).expand(B, self.num_samples, -1).reshape(
                    B * self.num_samples, -1
                )
            ).view(B, self.num_samples).mean(1, keepdim=True)
        ).squeeze(1)  # B,
        a1_given_z1_dist = D.multivariate_normal.MultivariateNormal(
            z1_given_zeta_mean @ self.observation_matrix + self.observation_offset,
            self.observation_covariance,
        )
        kl_a1 = (
            auxiliary_inference_log_likelihood[:, 0, :]
            - a1_given_z1_dist.log_prob(auxiliary_samples[:, 0, :])
        )
        
        kl_at = []
        kl_zt_minus_1 = []
        for t in range(1, T):
            zt_minus_1_filtered_dist = D.multivariate_normal.MultivariateNormal(
                filtered_z_mean[:, t - 1, :], filtered_z_cov[:, t - 1, :, :]
            )
            zt_minus_1_smoothed_dist = D.multivariate_normal.MultivariateNormal(
                smoothed_z_mean[:, t - 1, :], smoothed_z_cov[:, t - 1, :, :]
            )
            zt_minus_1_samples = zt_minus_1_smoothed_dist.rsample()  # shape (B, D_z)
            zt_given_zt_minus_1_mean = transition_matrices[t-1] @ zt_minus_1_samples + control_matrices[t-1] @ control[:, t-1, :]
            zt_given_zt_minus_1_cov = transition_covariances[t-1]
            at_given_zt_minus_1_mean = zt_given_zt_minus_1_mean @ self.observation_matrix + self.observation_offset
            at_given_zt_minus_1_cov = self.observation_matrix.T @ zt_given_zt_minus_1_cov @ self.observation_matrix + self.observation_covariance
            at_given_zt_minus_1_dist = D.multivariate_normal.MultivariateNormal(
                at_given_zt_minus_1_mean, at_given_zt_minus_1_cov
            )
            kl_at = (
                auxiliary_inference_log_likelihood[:, t, :]
                - at_given_zt_minus_1_dist.log_prob(auxiliary_samples[:, t, :])
            )
            kl_zt_minus_1 = (
                zt_minus_1_smoothed_dist.log_prob(zt_minus_1_samples)
                - zt_minus_1_filtered_dist.log_prob(zt_minus_1_samples)
            )
            kl_at.append(kl_at)
            kl_zt_minus_1.append(kl_zt_minus_1)
        kl_at = torch.stack(kl_at, dim=1).sum(1)  # shape (B, )
        kl_zt_minus_1 = torch.stack(kl_zt_minus_1, dim=1).sum(1)  # shape (B, )
        rate = kl_zeta + kl_z1 + kl_a1 + kl_at + kl_zt_minus_1
        return rate

    def get_loss(self, distortion, rate):
        pass
