import torch
import torch.nn as nn
from subnetworks import InitialStateNetwork, ContinuousTransition, AuxiliaryInferenceNetwork, AuxiliaryObservationNetwork

class EKVAE(nn.Module):
    '''
    The EKVAE model without constrained optimization framework.
    '''
    def __init__(self, continuous_dim, auxiliary_dim, hidden_dim, num_samples, num_base_matrices):
        super(EKVAE, self).__init__()
        self.initial_state_network = InitialStateNetwork(continuous_dim, hidden_dim)
        self.continuous_transition = ContinuousTransition(continuous_dim, num_base_matrices, hidden_dim)
        self.auxiliary_inference_network = AuxiliaryInferenceNetwork(continuous_dim, hidden_dim)
        self.auxiliary_observation_network = AuxiliaryObservationNetwork(continuous_dim, auxiliary_dim, hidden_dim)
    def forward(self, x):
        B, T, D = x.shape
        a_mean, a_cov = self.auxiliary_inference_network(x)
        # create a normal dist with length B, T
        a_dist = D.normal.Normal(a_mean, a_cov).to_event(1)
        a = a_dist.rsample()
        
        q_a_x = a_dist.log_prob(a) # log q(a_t | x_t)
        # entropy of q(a_t | x_t)
        entropy_q_a_x = -a_dist.entropy()

        z_1_mean, z_1_covariance = self.initial_state_network(B) # Samples z_1 from a standard normal distribution
        self.kalman_filter(a, z_1_mean, z_1_covariance)
        self.kalman_smoother(a, z_1_mean, z_1_covariance)

    def kalman_filter(self, a, z_1_mean, z_1_covariance):
        B, T, D = a.shape
        z_t_mean = z_1_mean
        z_t_covariance = z_1_covariance
        for t in range(T):
            z_t_mean, z_t_covariance = self.kalman_forward_step(a[:, t, :], z_t_mean, z_t_covariance)
        return z_t_mean, z_t_covariance

    def kalman_forward_step(self, a_t, z_t_mean, z_t_covariance):
        # can one leverage JIT to speed up the kalman filter?
        A = a_t.shape[-1]
        D = z_t_mean.shape[-1]
        # Prediction step
        mean_transition, covariance_transition = self.continuous_transition(z_t_mean, z_t_covariance)
        z_t_pred_mean = mean_transition @ z_t_mean
        z_t_pred_covariance = mean_transition @ z_t_covariance @ mean_transition.T + covariance_transition
        
        # Update step
        mean_update = self.auxiliary_inference_network.H @ z_t_pred_mean
        S = z_t_pred_covariance[:, :A, :A] + self.auxiliary_observation_network.R
        K = z_t_pred_covariance[:, :A] @ torch.inverse(S)
        z_t_mean = z_t_pred_mean + K @ (a_t - mean_update)
        z_t_covariance = z_t_pred_covariance - K @ S @ K.T
        return z_t_mean, z_t_covariance