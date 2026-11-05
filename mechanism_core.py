"""
Module: mechanism_core.py
Responsibility: Manages sparse stoichiometry structures and executes 
differentiable rate laws in log space (KPPax Algorithm 2).
"""

import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field
from typing import Tuple, List
from functools import partial

# Ensure 64-bit precision is enabled for stiff chemistry calculations
jax.config.update("jax_enable_x64", True)


class MechanismConfig(BaseModel):
    """
    Pydantic schema to validate chemical mechanism inputs before JAX execution.
    """
    n_species: int = Field(..., description="Total number of chemical species")
    n_reactions: int = Field(..., description="Total number of chemical reactions")
    species_names: List[str] = Field(..., description="Names of all chemical species")
    
    class Config:
        arbitrary_types_allowed = True


# --- Standalone JIT-Compiled Function with Static Arguments ---
# static_argnums=(4, 5) tells JAX that n_species and n_reactions are static integers, not tracers.
@partial(jax.jit, static_argnums=(4, 5))
def evaluate_log_space_rate_law(y: jnp.ndarray, 
                                k: jnp.ndarray, 
                                s_minus_indices: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                                s_net_indices: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                                n_species: int, 
                                n_reactions: int) -> jnp.ndarray:
    """
    Evaluates the rate law in log space to maintain full automatic 
    differentiation compatibility in JAX (KPPax Algorithm 2)[cite: 4].
    """
    # 1. Clip concentrations to a small positive floor to prevent log(0) errors[cite: 5]
    y_floored = jnp.maximum(y, 1e-30)
    
    # 2. Gather and scale log-concentrations by partial stoichiometric orders[cite: 4]
    i_r, j_r, v_r = s_minus_indices
    q = v_r * jnp.log(y_floored[i_r])
    
    # 3. Scatter-add onto the reaction axis (log product of reactants)[cite: 4]
    log_prod = jax.ops.segment_sum(q, j_r, num_segments=n_reactions)
    
    # 4. Recover reaction rates by exponentiating and multiplying by rate coefficients[cite: 4]
    omega = k * jnp.exp(log_prod)
    
    # 5. Scatter-add onto species axis to compute concentration time derivatives[cite: 3, 4]
    i_s, j_s, v_s = s_net_indices
    dy_dt = jax.ops.segment_sum(v_s * omega[j_s], i_s, num_segments=n_species)
    
    return dy_dt


class LogSpaceRateEngine:
    """
    Object-oriented wrapper that holds static indices and exposes 
    clean evaluation methods.
    """
    def __init__(self, config: MechanismConfig,
                 s_minus_indices: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 s_net_indices: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
        self.config = config
        self.s_minus_indices = s_minus_indices
        self.s_net_indices = s_net_indices

    def evaluate(self, y: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the JIT-compiled rate law using instance configuration attributes.
        """
        return evaluate_log_space_rate_law(
            y, k, 
            self.s_minus_indices, 
            self.s_net_indices, 
            self.config.n_species, 
            self.config.n_reactions
        )


# --- Unit Test ---
if __name__ == "__main__":
    print("[+] Testing Refactored Mechanism Core Module...")
    
    config = MechanismConfig(
        n_species=3,
        n_reactions=2,
        species_names=["A", "B", "C"]
    )
    
    s_minus = (jnp.array([0, 1]), jnp.array([0, 1]), jnp.array([1.0, 1.0]))
    s_net = (jnp.array([0, 1, 2]), jnp.array([0, 1, 1]), jnp.array([-1.0, -1.0, 1.0]))
    
    engine = LogSpaceRateEngine(config=config, s_minus_indices=s_minus, s_net_indices=s_net)
    
    y_test = jnp.array([1.2, 0.4, 0.1])
    k_test = jnp.array([0.2, 0.05])
    
    derivs = engine.evaluate(y_test, k_test)
    print(f"[+] Success! Computed dy/dt derivatives: {derivs}")