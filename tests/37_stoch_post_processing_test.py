"""
Test 37: plots of the stochastic reactors post-processing (StochDatabase, post_processing.py).

On the H2 stochastic database of the session fixture (h2_stoch_database.py).
Check:
- plot_T_Yk_indiv plots the 1D premixed flame (attributes of compute_1D_premixed, set here without computing it);
- plot_animation writes the animation in save_path when given;
- plot_indiv_traj writes one file per inlet and variable.
"""
import os

import numpy as np

from ai_reacting_flows.stochastic_reactors_data_gen.post_processing import StochDatabase


def test_stoch_post_processing_plots(h2_stoch_dtb, tmp_path):
    save_folder = str(tmp_path / "plots")
    stoch_dtb = StochDatabase(str(h2_stoch_dtb / "STOCH_DTB_TEST"), save_folder, with_traj=False)

    # 1D premixed flame, as set by compute_1D_premixed
    stoch_dtb.add_1D_premixed_archetype = True
    stoch_dtb.T_cano_1D = np.array([300.0, 1800.0])
    stoch_dtb.Y_cano_dict_1D = {"H2": np.array([0.02, 0.0])}
    stoch_dtb.plot_T_Yk_indiv(["H2"], iteration=3)
    assert os.path.isfile(os.path.join(save_folder, "dtb_T_H2_plot_00003.png"))

    # Animation in save_path
    save_path = str(tmp_path / "animation.gif")
    stoch_dtb.plot_animation("Temperature", "H2O", "Time", iterations=[0, 1], save_path=save_path)
    assert os.path.isfile(save_path)
    assert not os.path.isfile(os.path.join(save_folder, "dtb_animation.mp4"))

    # Trajectories of the particles: one file per inlet and variable
    stoch_dtb.plot_indiv_traj(1, "Temperature")
    stoch_dtb.plot_indiv_traj(2, "H2O")
    assert os.path.isfile(os.path.join(save_folder, "indiv_trajectory_inlet1_Temperature.png"))
    assert os.path.isfile(os.path.join(save_folder, "indiv_trajectory_inlet2_H2O.png"))
