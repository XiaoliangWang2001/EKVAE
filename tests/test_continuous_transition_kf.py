import torch
import torch.nn as nn
import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.subnetworks import KFContinuousTransition

class SimpleMixtureNetwork(nn.Module):
    """Simple mixture network for testing"""
    def __init__(self, state_dim, control_dim, num_matrices):
        super().__init__()
        self.net = nn.Linear(state_dim + control_dim, num_matrices)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, z, u):
        combined = torch.cat([z, u], dim=-1)
        return self.softmax(self.net(combined))

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def test_dimensions():
    return {
        'batch_size': 32,
        'seq_length': 10,
        'state_dim': 3,
        'control_dim': 2,
        'obs_dim': 3,
        'num_matrices': 4,
        'hidden_dim': 64
    }

@pytest.fixture
def test_model(test_dimensions, device):
    mixture_network = SimpleMixtureNetwork(
        state_dim=test_dimensions['state_dim'],
        control_dim=test_dimensions['control_dim'],
        num_matrices=test_dimensions['num_matrices']
    ).to(device)
    
    model = KFContinuousTransition(
        continuous_dim=test_dimensions['state_dim'],
        control_dim=test_dimensions['control_dim'],
        num_base_matrices=test_dimensions['num_matrices'],
        mixture_network=mixture_network
    ).to(device)
    
    return model

@pytest.fixture
def test_data(test_dimensions, device):
    d = test_dimensions
    
    # Generate random test data
    data = {
        'observations': torch.randn(d['batch_size'], d['seq_length'], d['obs_dim']),
        'controls': torch.randn(d['batch_size'], d['seq_length'], d['control_dim']),
        'observation_matrices': torch.eye(d['obs_dim'], d['state_dim']).unsqueeze(0).unsqueeze(0).expand(
            d['batch_size'], d['seq_length'], -1, -1
        ),
        'observation_covariance': torch.eye(d['obs_dim']).unsqueeze(0).unsqueeze(0).expand(
            d['batch_size'], d['seq_length'], -1, -1
        ),
        'observation_offsets': torch.zeros(d['batch_size'], d['seq_length'], d['obs_dim']),
        'initial_state_mean': torch.zeros(d['batch_size'], d['state_dim']),
        'initial_state_covariance': torch.eye(d['state_dim']).unsqueeze(0).expand(d['batch_size'], -1, -1)
    }
    
    # Move all tensors to device
    return {k: v.to(device) for k, v in data.items()}

def test_forward_pass(test_model, test_data):
    """Test if forward pass runs without errors and outputs have correct shapes"""
    model = test_model
    
    # Test filter mode
    filtered_means, filtered_covs = model(**test_data, mode="filter")
    
    assert filtered_means.shape == (
        test_data['observations'].shape[0],  # batch
        test_data['observations'].shape[1],  # sequence
        test_data['initial_state_mean'].shape[-1]  # state dim
    )
    
    assert filtered_covs.shape == (
        test_data['observations'].shape[0],  # batch
        test_data['observations'].shape[1],  # sequence
        test_data['initial_state_mean'].shape[-1],  # state dim
        test_data['initial_state_mean'].shape[-1]  # state dim
    )
    
    # Test smooth mode
    smoothed_means, smoothed_covs = model(**test_data, mode="smooth")
    assert smoothed_means.shape == filtered_means.shape
    assert smoothed_covs.shape == filtered_covs.shape

def test_gradients(test_model, test_data, device):
    """Test if gradients flow through the model properly"""
    model = test_model
    
    # Enable gradient tracking for observations
    observations = test_data['observations'].clone().requires_grad_(True)
    test_data['observations'] = observations
    
    # Forward pass
    filtered_means, filtered_covs = model(**test_data, mode="smooth")
    
    # Compute loss (simple MSE loss for testing)
    loss = filtered_means.pow(2).mean()
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist and are not None
    assert observations.grad is not None
    assert any(p.grad is not None for p in model.parameters())
    
    # Check if gradients have valid values (not NaN or inf)
    assert not torch.isnan(observations.grad).any()
    assert not torch.isinf(observations.grad).any()
    
    for param in model.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

def test_training_inference_modes(test_model, test_data):
    """Test if model behaves differently in training and inference modes"""
    model = test_model
    
    # Training mode
    model.train()
    with torch.set_grad_enabled(True):
        train_means1, _ = model(**test_data, mode="filter")
        train_means2, _ = model(**test_data, mode="filter")
        
        # In training mode, outputs should be different due to sampling
        assert not torch.allclose(train_means1, train_means2)
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        eval_means1, _ = model(**test_data, mode="filter")
        eval_means2, _ = model(**test_data, mode="filter")
        
        # In eval mode, outputs should be deterministic
        assert torch.allclose(eval_means1, eval_means2)

def test_numerical_stability(test_model, test_data):
    """Test numerical stability with extreme values"""
    model = test_model
    
    # Test with very large values
    large_obs = test_data['observations'] * 1e6
    test_data_large = test_data.copy()
    test_data_large['observations'] = large_obs
    
    filtered_means, filtered_covs = model(**test_data_large, mode="filter")
    assert not torch.isnan(filtered_means).any()
    assert not torch.isnan(filtered_covs).any()
    assert not torch.isinf(filtered_means).any()
    assert not torch.isinf(filtered_covs).any()
    
    # Test with very small values
    small_obs = test_data['observations'] * 1e-6
    test_data_small = test_data.copy()
    test_data_small['observations'] = small_obs
    
    filtered_means, filtered_covs = model(**test_data_small, mode="filter")
    assert not torch.isnan(filtered_means).any()
    assert not torch.isnan(filtered_covs).any()
    assert not torch.isinf(filtered_means).any()
    assert not torch.isinf(filtered_covs).any() 