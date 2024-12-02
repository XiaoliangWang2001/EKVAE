from dataclasses import dataclass
from typing import Literal
from pathlib import Path
import yaml

@dataclass
class NetworkConfig:
    hidden_dim: int
    num_layers: int
    dropout: float = 0.1


@dataclass
class InitialStateConfig:
    hidden_dim: int

@dataclass
class AuxiliaryConfig:
    hidden_dim: int

@dataclass
class KalmanConfig:
    num_base_matrices: int
    compile_mode: bool = True
    observation_noise: float = 1.0
    mixture_net_dim: int = 64
@dataclass
class EKVAEConfig:
    continuous_dim: int
    control_dim: int
    auxiliary_dim: int
    num_samples: int
    strategy: Literal["CO", "Anneal", "Vanilla"]
    network: NetworkConfig
    kalman: KalmanConfig
    initial_state: InitialStateConfig
    auxiliary: AuxiliaryConfig
    @classmethod
    def from_yaml(cls, yaml_path: str | Path, validate: bool = True):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        initial_state_config = InitialStateConfig(**config_dict.pop('initial_state'))
        network_config = NetworkConfig(**config_dict.pop('network'))
        kalman_config = KalmanConfig(**config_dict.pop('kalman'))
        auxiliary_config = AuxiliaryConfig(**config_dict.pop('auxiliary'))
        instance = cls(
            **config_dict,
            network=network_config,
            kalman=kalman_config,
            initial_state=initial_state_config,
            auxiliary=auxiliary_config
        )
        
        return instance.validate() if validate else instance

    def to_yaml(self, yaml_path: str | Path):
        config_dict = {
            'continuous_dim': self.continuous_dim,
            'control_dim': self.control_dim,
            'auxiliary_dim': self.auxiliary_dim,
            'num_samples': self.num_samples,
            'strategy': self.strategy,
            'network': self.network.__dict__,
            'kalman': self.kalman.__dict__,
            'initial_state': self.initial_state.__dict__,
            'auxiliary': self.auxiliary.__dict__
        }
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        return self.validate()  # Validate after updates
    

    def validate(self):
        if self.continuous_dim <= self.auxiliary_dim:
            raise ValueError("Continuous dimension must be greater than auxiliary dimension")
