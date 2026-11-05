"""
Module: solver_module.py
Responsibility: Wraps diffrax to perform stiff numerical integration and time-stepping
for chemical kinetics equations evaluated in log-space.
"""

import jax
import jax.numpy as jnp
import diffrax
from pydantic import BaseModel, Field
from typing import Tuple, List

# Import your Phase 1 core engine module
from mechanism_core import MechanismConfig, evaluate_log_space_rate_law

# Enable 64-bit precision for numerical stability during stiff integration
jax.config.update("jax_enable_x64", True)


class SolverConfig(BaseModel):
    """
    Pydantic schema to validate ODE solver hyperparameters.
    """
    rtol: float = Field(1e-8, description="Relative error tolerance")
    atol: float = Field(1e-9, description="Absolute error tolerance")
    dt0: float = Field(1e-4, description="Initial integration step size in minutes")
    max_steps: int = Field(10000, description="Maximum solver steps allowed")

    class Config:
        arbitrary_types_allowed = True


class StiffChemistrySolver:
    """
    High-level interface for running time-based chemical simulations using diffrax.
    """
    def __init__(self, mech_config: MechanismConfig,
                 s_minus_indices: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 s_net_indices: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 solver_config: SolverConfig = SolverConfig()):
        self.mech_config = mech_config
        self.s_minus_indices = s_minus_indices
        self.s_net_indices = s_net_indices
        self.solver_config = solver_config

    def simulate(self, y0: jnp.ndarray, k: jnp.ndarray, duration_minutes: float) -> jnp.ndarray:
        """
        Runs the simulation starting from initial concentrations y0 over duration_minutes.
        """
        t0, t1 = 0.0, float(duration_minutes)
        
        # Define the Right-Hand-Side (RHS) differential function
        def rhs(t, y, args):
            rate_constants = args
            return evaluate_log_space_rate_law(
                y, rate_constants, 
                self.s_minus_indices, 
                self.s_net_indices, 
                self.mech_config.n_species, 
                self.mech_config.n_reactions
            )

        term = diffrax.ODETerm(rhs)
        
        # Kvaerno3 solver with a PID step-size controller handles severe chemical stiffness
        solver = diffrax.Kvaerno3()
        stepsize_controller = diffrax.PIDController(
            rtol=self.solver_config.rtol, 
            atol=self.solver_config.atol
        )
        
        # diffrax handles compilation and stepping natively
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=self.solver_config.dt0,
            y0=y0,
            args=k,
            stepsize_controller=stepsize_controller,
            max_steps=self.solver_config.max_steps,
            adjoint=diffrax.RecursiveCheckpointAdjoint()  # Memory-efficient reverse-mode gradients
        )
        
        return solution.ys[-1]


# --- Runnable Interactive Unit Test ---
if __name__ == "__main__":
    print("[+] Testing Phase 2: Stiff ODE Solver Engine...")

    # Configure a mock mechanism (3 species: A, B, C | 2 reactions)
    mech_conf = MechanismConfig(
        n_species=3,
        n_reactions=2,
        species_names=["Species A", "Species B", "Species C"]
    )

    s_minus = (jnp.array([0, 1]), jnp.array([0, 1]), jnp.array([1.0, 1.0]))
    s_net = (jnp.array([0, 1, 2]), jnp.array([0, 1, 1]), jnp.array([-1.0, -1.0, 1.0]))

    solver = StiffChemistrySolver(mech_config=mech_conf, s_minus_indices=s_minus, s_net_indices=s_net)

    print("\n--- ENTER SIMULATION INPUTS ---")
    try:
        init_a = float(input("Enter initial concentration for Species A (e.g., 1.0): ").strip())
        init_b = float(input("Enter initial concentration for Species B (e.g., 0.5): ").strip())
        init_c = float(input("Enter initial concentration for Species C (e.g., 0.3): ").strip())
        sim_time = float(input("Enter simulation duration in minutes (e.g., 60): ").strip())

        y_initial = jnp.array([init_a, init_b, init_c])
        k_constants = jnp.array([0.1, 0.05])  # Rate constants

        print(f"\n[-] Running simulation for {sim_time} minutes...")
        y_final = solver.simulate(y_initial, k_constants, duration_minutes=sim_time)

        print("\n--- SIMULATION FINAL RESULTS ---")
        for i, name in enumerate(mech_conf.species_names):
            print(f"Initial {name}: {y_initial[i]:.4f}  --->  Final {name} at t={sim_time}min: {y_final[i]:.4f}")

    except Exception as e:
        print(f"[!] Error during execution: {e}")