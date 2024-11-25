import torch
import torch.nn as nn

from ..torch_kalman import specialized_standard
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
        pass