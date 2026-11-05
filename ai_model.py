"""
Module: ai_model.py
Responsibility: Implements a JAX/Equinox neural network to predict dynamic 
emission drivers and rate coefficient corrections for the chemistry solver.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from pydantic import BaseModel, Field
from typing import List

# Ensure 64-bit precision is enabled
jax.config.update("jax_enable_x64", True)


class AIModelConfig(BaseModel):
    """
    Pydantic schema to validate neural network architecture hyperparameters.
    """
    input_dim: int = Field(4, description="Number of environmental input features (e.g., Temp, Sunlight, Humidity, Traffic)")
    hidden_dim: int = Field(32, description="Hidden layer width")
    output_dim: int = Field(2, description="Output dimension (e.g., rate multipliers for our 2 reactions)")
    depth: int = Field(2, description="Number of hidden layers")

    class Config:
        arbitrary_types_allowed = True


class EmissionCorrectionNN(eqx.Module):
    """
    A physics-informed neural network component built using Equinox.
    It predicts dynamic rate coefficient corrections based on environmental states.
    """
    layers: List[eqx.Module]

    def __init__(self, config: AIModelConfig, key: jax.random.PRNGKey):
        # Split random keys for weight initialization across layers
        keys = jax.random.split(key, config.depth + 1)
        
        self.layers = []
        
        # Input layer
        self.layers.append(eqx.nn.Linear(config.input_dim, config.hidden_dim, key=keys[0]))
        self.layers.append(jax.nn.silu)  # Smooth SiLU activation function for gradient stability
        
        # Hidden layers
        for i in range(config.depth - 1):
            self.layers.append(eqx.nn.Linear(config.hidden_dim, config.hidden_dim, key=keys[i + 1]))
            self.layers.append(jax.nn.silu)
            
        # Output layer (mapped with softplus or exponential to ensure positive rate multipliers)
        self.layers.append(eqx.nn.Linear(config.hidden_dim, config.output_dim, key=keys[-1]))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass: Maps environmental features to positive rate multipliers.
        """
        for layer in self.layers:
            x = layer(x)
            
        # Use softplus to guarantee strictly positive rate multipliers (preventing unphysical negative rates)
        return jnn_positive_scaler(x)


@jax.jit
def jnn_positive_scaler(x: jnp.ndarray) -> jnp.ndarray:
    """Ensures AI-predicted multipliers remain strictly positive using softplus."""
    return jax.nn.softplus(x) + 1e-3


# --- Runnable Unit Test & Verification ---
if __name__ == "__main__":
    print("[+] Testing Phase 3: AI Neural Network Model...")

    # 1. Define model config
    config = AIModelConfig(
        input_dim=4,
        hidden_dim=16,
        output_dim=2, # Matches our 2 reactions from earlier phases
        depth=2
    )

    # 2. Initialize random number generator key for JAX
    rng_key = jax.random.PRNGKey(42)

    # 3. Instantiate the neural network model
    ai_model = EmissionCorrectionNN(config, key=rng_key)

    # 4. Create mock environmental inputs (e.g., [Temperature, Sunlight, Humidity, Traffic Index])
    environmental_features = jnp.array([298.15, 0.85, 0.60, 1.25])

    # 5. Run forward pass prediction
    predicted_multipliers = ai_model(environmental_features)

    print("\n--- AI PREDICTION RESULTS ---")
    print(f"Input Environmental Features: {environmental_features}")
    print(f"AI-Predicted Rate Multipliers: {predicted_multipliers}")
    print("[+] Success! Neural network initialized and executed forward pass successfully.")