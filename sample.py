"""
Module: smog_predictor_app.py
Responsibility: Interactive user-facing application that takes environmental inputs,
runs the physics-AI simulation, and outputs a human-readable smog forecast.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

# Import your core modules
from mechanism_core import MechanismConfig
from solver_module import StiffChemistrySolver
from ai_model import AIModelConfig, EmissionCorrectionNN

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

def run_neighborhood_forecast():
    print("==================================================")
    print("   SMART-CITY HYPER-LOCAL SMOG FORECASTER (v1.0)")
    print("==================================================\n")

    # 1. Get User Input (Simulating virtual sensor inputs)
    print("Enter current neighborhood environmental conditions:")
    try:
        temp_c = float(input(" -> Temperature (°C, e.g., 32.5): ").strip())
        temp_k = temp_c + 273.15  # Convert to Kelvin for physics engine
        
        sunlight = float(input(" -> Sunlight Intensity Index (0.0 to 1.0, e.g., 0.85): ").strip())
        humidity = float(input(" -> Humidity Level (0.0 to 1.0, e.g., 0.60): ").strip())
        traffic = float(input(" -> Traffic Congestion Index (e.g., 1.4 for heavy rush hour): ").strip())
        
    except ValueError:
        print("[!] Invalid input. Please enter numeric values.")
        return

    # Pack into feature array for the AI model
    env_features = jnp.array([temp_k, sunlight, humidity, traffic])

    # 2. Initialize the Physics-AI Engine behind the scenes
    mech_conf = MechanismConfig(
        n_species=3, 
        n_reactions=2, 
        species_names=["Nitric Oxide (NO)", "Nitrogen Dioxide (NO2)", "Ground-Level Ozone (O3)"]
    )
    s_minus = (jnp.array([0, 1]), jnp.array([0, 1]), jnp.array([1.0, 1.0]))
    s_net = (jnp.array([0, 1, 2]), jnp.array([0, 1, 1]), jnp.array([-1.0, -1.0, 1.0]))

    solver = StiffChemistrySolver(mech_conf, s_minus_indices=s_minus, s_net_indices=s_net)
    
    ai_config = AIModelConfig(input_dim=4, hidden_dim=16, output_dim=2, depth=2)
    ai_model = EmissionCorrectionNN(ai_config, key=jax.random.PRNGKey(42))
    
    k_baseline = jnp.array([0.1, 0.05])

    # 3. Run the AI + Physics Simulation for a 60-minute forecast window
    print("\n[*] Running physics-constrained AI simulation...")
    
    # AI predicts rate adjustments based on your inputs
    multipliers = ai_model(env_features)
    k_adjusted = k_baseline * multipliers
    
    # Starting baseline concentrations of pollutants in the neighborhood
    y0 = jnp.array([1.0, 0.5, 0.3]) 
    
    # Solve stiff ODE forward in time (60 minutes)
    forecast_result = solver.simulate(y0, k_adjusted, duration_minutes=60.0)

    no_conc, no2_conc, o3_conc = float(forecast_result[0]), float(forecast_result[1]), float(forecast_result[2])

    # 4. Translate Raw Math into User-Friendly Output & Risk Evaluation
    print("\n--------------------------------------------------")
    print("               AIR QUALITY FORECAST REPORT         ")
    print("--------------------------------------------------")
    print(f"Forecast Horizon : 60 Minutes Ahead")
    print(f"Predicted Pollutant Levels:")
    print(f" • Nitric Oxide (NO)          : {no_conc:.4f} ppm")
    print(f" • Nitrogen Dioxide (NO2)     : {no2_conc:.4f} ppm")
    print(f" • Ground-Level Ozone (O3)    : {o3_conc:.4f} ppm")

    # Smart threshold alert logic
    print("\n[+] Health & Safety Assessment:")
    if o3_conc > 0.40 or no2_conc > 0.40:
        print(" ⚠️  ALERT: High smog risk detected! Ozone/NO2 levels are spiking.")
        print("     Recommendation: Sensitive groups should avoid outdoor exercise.")
    elif o3_conc > 0.20:
        print(" ⚡ CAUTION: Moderate pollution buildup. Air quality is acceptable,")
        print("     but traffic corridors may experience light haze.")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    run_neighborhood_forecast()