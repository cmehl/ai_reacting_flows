"""Builds dtb_params.yaml (full run) and smoke/dtb_params.yaml (tiny test) for the rollout stochastic reactor,
starting from the CURL_MODIFIED reduced-mechanism baseline params (STOCH_REDUCED/dtb_params.yaml)."""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))

s = open(f"{HERE}/../STOCH_REDUCED/dtb_params.yaml").read()
s = s.replace("mech_file: ../../data/Giovanni/STEC_A_noAR.yaml", "mech_file: ./STEC_A_noAR.yaml")
# Variable marching dt (coarse while cold/quasi-steady mixing, fine through ignition) ...
s = s.replace("time_step: 5.0e-6\n", "time_step:\n  0.0: 2.0e-5\n  0.006: 5.0e-6\nrollout_steps: 3\nrollout_dt: 5.0e-6\n")
s = s.replace("time_max: 0.05", "time_max: 0.0125")
s = s.replace("results_folder_suffix: NH3_H2_N2", "results_folder_suffix: NH3_H2_N2_ROLLOUT")
s = s.replace("automatic_termination: true", "automatic_termination: false")
open(f"{HERE}/dtb_params.yaml", "w").write(s)

t = s.replace("0.006: 5.0e-6", "0.0002: 5.0e-6").replace("time_max: 0.0125", "time_max: 0.0004").replace("NH3_H2_N2_ROLLOUT", "SMOKE")
for a, b in [("nb_particles: 1680", "nb_particles: 42"), ("nb_particles: 308", "nb_particles: 8"),
             ("nb_particles: 12", "nb_particles: 2"), ("nb_particles: 400", "nb_particles: 10")]:
    t = t.replace(a, b)
os.makedirs(f"{HERE}/smoke", exist_ok=True)
open(f"{HERE}/smoke/dtb_params.yaml", "w").write(t)
for f in ("STEC_A_noAR.yaml", "generate_stoch_dtb.py"):
    shutil.copy(f"{HERE}/{f}", f"{HERE}/smoke/{f}")
