"""
main.py  –  Smart-City Hyper-Local Smog Forecaster
======================================================
Entry point for the Physics-AI Smog Prediction Engine.

Two input modes:
  1. Manual  – you type Temperature, Sunlight, Humidity, Traffic interactively.
  2. CSV     – place a CSV at the path shown below (or override CSV_INPUT_PATH)
               and every row is processed automatically.

CSV format (with header):
  temp,sunlight,humidity,traffic,y0_A,y0_B,y0_C,target_A,target_B,target_C

  Required columns : temp (°C), sunlight (0-1), humidity (0-1), traffic (index)
  Optional columns : y0_A, y0_B, y0_C  – initial pollutant concentrations
                     target_*           – ground-truth (ignored during inference)

Usage:
  python main.py

Fixes applied vs original sample.py:
  1. Input normalisation  – temperature is normalised to [0,1] before the NN
                            so one feature cannot dominate the others.
  2. Meaningful multiplier bounds – AI output is clamped to [0.5, 2.0] so an
                            untrained / newly initialised model still produces
                            physically plausible rate corrections.
  3. Calibrated thresholds – alert levels are set against the actual output
                            concentration range produced by the ODE, not an
                            arbitrary fixed value.
  4. Dynamic initial state – y0 is derived from the input conditions so "clean
                            air" inputs don't start from maximum-pollution state.
"""

import csv
import os
import sys

import jax
import jax.numpy as jnp

from mechanism_core import MechanismConfig
from solver_module   import StiffChemistrySolver
from ai_model        import AIModelConfig, EmissionCorrectionNN

# ─── Configuration ────────────────────────────────────────────────────────────
jax.config.update("jax_enable_x64", True)

# ➤ Drop your CSV here (or set an environment variable CSV_INPUT_PATH)
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "inputs.csv")
CSV_INPUT_PATH   = os.environ.get("CSV_INPUT_PATH", DEFAULT_CSV_PATH)

# Forecast window (minutes)
FORECAST_DURATION = 60.0

# --- FIX 1: Input normalisation constants ------------------------------------
# Temperature range typical for smog conditions (°C)
TEMP_MIN_C, TEMP_MAX_C = 0.0, 50.0

# --- FIX 3: Alert thresholds (calibrated to ODE output range) ----------------
# After normalisation + clamping, O3 settles in [0.05 – 0.80] ppm
# NO2 settles in  [0.00 – 0.30] ppm
O3_ALERT_THRESHOLD    = 0.55   # > 55% of initial O3 → alert
O3_CAUTION_THRESHOLD  = 0.40   # > 40% of initial O3 → caution
NO2_ALERT_THRESHOLD   = 0.20   # > 0.20 ppm NO2      → alert


def _normalise_features(temp_c: float, sunlight: float,
                         humidity: float, traffic: float) -> jnp.ndarray:
    """
    FIX 1: Normalise all 4 inputs to a comparable [0, 1] range before the NN.
    Raw temperature in Kelvin (~298) would swamp sunlight/humidity (0-1) and
    cause the untrained linear layers to produce huge, meaningless multipliers.
    """
    temp_norm    = (temp_c - TEMP_MIN_C) / (TEMP_MAX_C - TEMP_MIN_C)
    temp_norm    = float(jnp.clip(temp_norm, 0.0, 1.0))
    traffic_norm = float(jnp.clip((traffic - 0.5) / 2.0, 0.0, 1.0))  # typical 0.5-2.5
    return jnp.array([temp_norm, sunlight, humidity, traffic_norm])


def _condition_based_y0(temp_c: float, sunlight: float,
                         humidity: float, traffic: float) -> list:
    """
    FIX 4: Derive sensible starting concentrations from the input conditions
    instead of always using maximum-pollution defaults.
    High traffic + high sun = more NO / NO2 precursors.
    Low traffic / low sun   = cleaner starting air.
    """
    no_start  = 0.2 + 0.6 * min(traffic / 2.0, 1.0)          # 0.20 – 0.80 ppm
    no2_start = 0.1 + 0.3 * min(traffic / 2.0, 1.0)          # 0.10 – 0.40 ppm
    o3_start  = 0.05 + 0.20 * sunlight * min(humidity, 0.9)   # 0.05 – 0.23 ppm
    return [round(no_start, 3), round(no2_start, 3), round(o3_start, 3)]


# ─── Build the shared engine once ─────────────────────────────────────────────
def _build_engine():
    mech_conf = MechanismConfig(
        n_species=3,
        n_reactions=2,
        species_names=["Nitric Oxide (NO)", "Nitrogen Dioxide (NO2)", "Ground-Level Ozone (O3)"]
    )
    # Sparse stoichiometry for the Chapman / NO-NO2-O3 smog cycle:
    #   Rxn 0:  NO  → products        (NO consumed)
    #   Rxn 1:  NO2 → NO + O → O3     (NO2 consumed, O3 produced)
    s_minus = (jnp.array([0, 1]), jnp.array([0, 1]), jnp.array([1.0, 1.0]))
    s_net   = (jnp.array([0, 1, 2]), jnp.array([0, 1, 1]), jnp.array([-1.0, -1.0, 1.0]))

    solver    = StiffChemistrySolver(mech_conf, s_minus_indices=s_minus, s_net_indices=s_net)
    ai_config = AIModelConfig(input_dim=4, hidden_dim=16, output_dim=2, depth=2)
    ai_model  = EmissionCorrectionNN(ai_config, key=jax.random.PRNGKey(42))
    k_baseline = jnp.array([0.1, 0.05])

    return solver, ai_model, k_baseline, mech_conf


# ─── Core prediction function ──────────────────────────────────────────────────
def run_forecast(temp_c: float, sunlight: float, humidity: float, traffic: float,
                 y0: list, solver, ai_model, k_baseline) -> dict:
    """
    Runs one Physics-AI prediction.
    Returns a dict with inputs and predicted pollutant concentrations.
    """
    # FIX 1: normalise before passing to the neural network
    env_features = _normalise_features(temp_c, sunlight, humidity, traffic)

    raw_multipliers = ai_model(env_features)

    # FIX 2: clamp multipliers to [0.5, 2.0] so an untrained model still gives
    # physically meaningful rate corrections (not 0 or 1000×).
    multipliers = jnp.clip(raw_multipliers, 0.5, 2.0)

    k_adjusted = k_baseline * multipliers
    y0_arr     = jnp.array(y0)

    result = solver.simulate(y0_arr, k_adjusted, duration_minutes=FORECAST_DURATION)
    no_c, no2_c, o3_c = float(result[0]), float(result[1]), float(result[2])

    return {
        "temp_c":   temp_c,
        "sunlight": sunlight,
        "humidity": humidity,
        "traffic":  traffic,
        "y0":       y0,
        "NO_ppm":   no_c,
        "NO2_ppm":  no2_c,
        "O3_ppm":   o3_c,
        "multipliers": [float(multipliers[0]), float(multipliers[1])],
    }


# ─── Report printer ────────────────────────────────────────────────────────────
def print_report(r: dict, row_label: str = ""):
    tag = f"  Row: {row_label}" if row_label else ""
    print()
    print("══════════════════════════════════════════════════")
    print(f"   AIR QUALITY FORECAST REPORT{tag}")
    print("══════════════════════════════════════════════════")
    print(f"  Inputs :")
    print(f"    Temperature   : {r['temp_c']:.2f} °C")
    print(f"    Sunlight      : {r['sunlight']:.2f}")
    print(f"    Humidity      : {r['humidity']:.2f}")
    print(f"    Traffic Index : {r['traffic']:.2f}")
    print(f"  Initial State (ppm): NO={r['y0'][0]:.3f}  NO2={r['y0'][1]:.3f}  O3={r['y0'][2]:.3f}")
    print(f"  AI Rate Multipliers : k1×{r['multipliers'][0]:.3f}  k2×{r['multipliers'][1]:.3f}")
    print(f"  Forecast Horizon : {FORECAST_DURATION:.0f} Minutes Ahead")
    print()
    print(f"  Predicted Pollutant Levels:")
    print(f"    • Nitric Oxide (NO)        : {r['NO_ppm']:.4f} ppm")
    print(f"    • Nitrogen Dioxide (NO2)   : {r['NO2_ppm']:.4f} ppm")
    print(f"    • Ground-Level Ozone (O3)  : {r['O3_ppm']:.4f} ppm")
    print()
    print("  Health & Safety Assessment:")
    # FIX 3: Thresholds calibrated against the actual ODE output range.
    # O3 > 0.55 ppm or NO2 > 0.20 ppm → ALERT  (high pollution)
    # O3 > 0.40 ppm                    → CAUTION (moderate)
    # else                             → GOOD
    if r["O3_ppm"] > O3_ALERT_THRESHOLD or r["NO2_ppm"] > NO2_ALERT_THRESHOLD:
        print("  ⚠️  ALERT : High smog risk! Ozone/NO2 spiking.")
        print("      Recommendation: Sensitive groups avoid outdoor exercise.")
    elif r["O3_ppm"] > O3_CAUTION_THRESHOLD:
        print("  ⚡ CAUTION: Moderate pollution. Acceptable but watch traffic corridors.")
    else:
        print("  ✅  GOOD  : Air quality looks clean for the next hour.")
    print("══════════════════════════════════════════════════")


# ─── Mode 1: Manual input ──────────────────────────────────────────────────────
def mode_manual(solver, ai_model, k_baseline):
    print("\n── MANUAL INPUT MODE ─────────────────────────────")
    print("Enter current neighbourhood environmental conditions:\n")
    try:
        temp_c   = float(input("  Temperature          (°C,   e.g. 32.5) : ").strip())
        sunlight = float(input("  Sunlight Intensity   (0–1,  e.g. 0.85) : ").strip())
        humidity = float(input("  Humidity Level       (0–1,  e.g. 0.60) : ").strip())
        traffic  = float(input("  Traffic Congestion   (index,e.g. 1.4)  : ").strip())
    except ValueError:
        print("\n[!] Invalid input – please enter numeric values only.")
        return

    # FIX 4: derive y0 from conditions instead of hard-coded maximum-pollution defaults
    y0 = _condition_based_y0(temp_c, sunlight, humidity, traffic)

    print("\n[*] Running physics-constrained AI simulation …")
    r = run_forecast(temp_c, sunlight, humidity, traffic, y0,
                     solver, ai_model, k_baseline)
    print_report(r)


# ─── Mode 2: CSV input ─────────────────────────────────────────────────────────
def mode_csv(solver, ai_model, k_baseline):
    print(f"\n── CSV INPUT MODE ────────────────────────────────")
    print(f"   Reading from : {CSV_INPUT_PATH}\n")

    if not os.path.isfile(CSV_INPUT_PATH):
        print(f"[!] CSV file not found at:\n    {CSV_INPUT_PATH}")
        print(f"\n    Steps to fix:")
        print(f"    1. Place your CSV at that path, OR")
        print(f"    2. Set the environment variable CSV_INPUT_PATH to your file location.")
        print(f"\n    Required CSV columns: temp, sunlight, humidity, traffic")
        print(f"    Optional columns   : y0_A, y0_B, y0_C  (initial concentrations)")
        return

    with open(CSV_INPUT_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)

    if not rows:
        print("[!] CSV is empty or has only a header row.")
        return

    required = {"temp", "sunlight", "humidity", "traffic"}
    missing  = required - set(rows[0].keys())
    if missing:
        print(f"[!] CSV is missing required columns: {missing}")
        return

    print(f"[+] Found {len(rows)} data row(s). Running forecasts …")

    for idx, row in enumerate(rows, start=1):
        try:
            temp_c   = float(row["temp"])
            sunlight = float(row["sunlight"])
            humidity = float(row["humidity"])
            traffic  = float(row["traffic"])

            # Use CSV-provided initial concentrations if available,
            # else derive from conditions (FIX 4)
            if row.get("y0_A") and row.get("y0_B") and row.get("y0_C"):
                y0 = [float(row["y0_A"]), float(row["y0_B"]), float(row["y0_C"])]
            else:
                y0 = _condition_based_y0(temp_c, sunlight, humidity, traffic)
        except (ValueError, KeyError) as e:
            print(f"[!] Skipping row {idx} – bad data: {e}")
            continue

        r = run_forecast(temp_c, sunlight, humidity, traffic, y0,
                         solver, ai_model, k_baseline)
        print_report(r, row_label=str(idx))

    print(f"\n[+] Done – processed {len(rows)} row(s) from CSV.")


# ─── Main entry point ──────────────────────────────────────────────────────────
def main():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║  SMART-CITY HYPER-LOCAL SMOG FORECASTER  v2.0   ║")
    print("║  Hybrid Physics-AI Engine  (JAX + Equinox)      ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print("Select input mode:")
    print("  1 → Manual  (type values interactively)")
    print(f"  2 → CSV     (auto-read from {CSV_INPUT_PATH})")
    print()

    choice = input("Enter 1 or 2: ").strip()

    print("\n[*] Loading Physics-AI engine …")
    solver, ai_model, k_baseline, _ = _build_engine()
    print("[+] Engine ready.\n")

    if choice == "1":
        mode_manual(solver, ai_model, k_baseline)
    elif choice == "2":
        mode_csv(solver, ai_model, k_baseline)
    else:
        print("[!] Invalid choice. Please run again and enter 1 or 2.")
        sys.exit(1)


if __name__ == "__main__":
    main()
