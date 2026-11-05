"""
Module: trainer.py
Responsibility: Implements the end-to-end training and optimization loop, 
connecting the AI model, Stiff ODE Solver, and Physics Core using JAX autodiff.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Tuple

# Import previous phases
from mechanism_core import MechanismConfig
from solver_module import StiffChemistrySolver, SolverConfig
from ai_model import AIModelConfig, EmissionCorrectionNN

# Ensure 64-bit precision is enabled
jax.config.update("jax_enable_x64", True)


class PhysicsAIPipeline(eqx.Module):
    """
    Combines the Neural Network and the Chemistry Solver into a single 
    differentiable end-to-end pipeline.
    """
    ai_model: EmissionCorrectionNN
    solver: StiffChemistrySolver
    k_baseline: jnp.ndarray

    def __init__(self, ai_model: EmissionCorrectionNN, solver: StiffChemistrySolver, k_baseline: jnp.ndarray):
        self.ai_model = ai_model
        self.solver = solver
        self.k_baseline = k_baseline

    def __call__(self, env_features: jnp.ndarray, y0: jnp.ndarray, duration: float) -> jnp.ndarray:
        """
        Forward pass: Environmental Data -> AI Multipliers -> Adjusted Rates -> Solver -> Final State
        """
        # 1. AI predicts rate multipliers from environmental conditions
        multipliers = self.ai_model(env_features)
        
        # 2. Scale baseline rate constants with AI predictions
        k_adjusted = self.k_baseline * multipliers
        
        # 3. Run physics-constrained time-stepping simulation
        y_final = self.solver.simulate(y0, k_adjusted, duration_minutes=duration)
        
        return y_final


def compute_loss(model: PhysicsAIPipeline, env_features: jnp.ndarray, y0: jnp.ndarray, 
                 duration: float, target_y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes Mean Squared Error (MSE) loss between the physics simulation and ground-truth sensor data.
    """
    pred_y = model(env_features, y0, duration)
    loss = jnp.mean((pred_y - target_y) ** 2)
    return loss, pred_y


# --- Runnable Training Unit Test ---
if __name__ == "__main__":
    print("[+] Initializing Phase 4: Training & Optimization Loop...")

    # 1. Setup Mechanism Configuration (3 species, 2 reactions)
    mech_conf = MechanismConfig(
        n_species=3,
        n_reactions=2,
        species_names=["Species A", "Species B", "Species C"]
    )
    s_minus = (jnp.array([0, 1]), jnp.array([0, 1]), jnp.array([1.0, 1.0]))
    s_net = (jnp.array([0, 1, 2]), jnp.array([0, 1, 1]), jnp.array([-1.0, -1.0, 1.0]))

    # 2. Setup Solver & AI Model
    solver = StiffChemistrySolver(mech_config=mech_conf, s_minus_indices=s_minus, s_net_indices=s_net)
    
    ai_config = AIModelConfig(input_dim=4, hidden_dim=16, output_dim=2, depth=2)
    rng_key = jax.random.PRNGKey(42)
    ai_model = EmissionCorrectionNN(ai_config, key=rng_key)
    
    k_baseline = jnp.array([0.1, 0.05])

    # 3. Build Full Pipeline
    pipeline = PhysicsAIPipeline(ai_model=ai_model, solver=solver, k_baseline=k_baseline)

    # 4. Define Mock Training Data (Environmental inputs, Initial state, Ground Truth target)
    env_sample = jnp.array([298.15, 0.85, 0.60, 1.25])
    y0_sample = jnp.array([1.0, 0.5, 0.3])
    simulation_duration = 60.0
    
    # Ground truth target concentrations we want the AI-physics model to learn to predict
    target_y_sample = jnp.array([0.003, 0.025, 0.772]) 

    # 5. Setup Optax Optimizer (Adam optimizer)
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(eqx.filter(pipeline, eqx.is_inexact_array))

    # Define training step function using Equinox gradient filtering
    @eqx.filter_jit
    def make_step(pipeline_model, opt_state, env, y0, duration, target):
        # Compute loss and gradients through the solver and AI model simultaneously
        (loss, preds), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(
            pipeline_model, env, y0, duration, target
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, pipeline_model)
        new_pipeline = eqx.apply_updates(pipeline_model, updates)
        return loss, preds, new_pipeline, new_opt_state

    print("\n--- STARTING TRAINING ITERATIONS ---")
    epochs = 5
    for epoch in range(epochs):
        loss_val, predictions, pipeline, opt_state = make_step(
            pipeline, opt_state, env_sample, y0_sample, simulation_duration, target_y_sample
        )
        print(f"Epoch {epoch + 1}/{epochs} | Loss (MSE): {loss_val:.6f}")

    print("\n[+] Success! End-to-end training loop executed successfully.")
    print(f"Final AI-predicted chemical concentrations after training: {predictions}")