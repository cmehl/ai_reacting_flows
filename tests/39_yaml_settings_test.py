"""
Test 39: settings read from the YAML inputs instead of being hardcoded.

Check:
- flamelets database (dtb_params_flmts.yaml): the fuel inlet of the diffusion flames is at oned_diff_params.T_fuel
  (lowest temperature of the flame, the air inlet being hotter), and the scatter plot of the database shows
  scatter_plot_variables;
- training (networks_params.yaml): ReduceLROnPlateau is built with learning.plateau_factor and plateau_patience.

Independent of the other tests.
"""
import os

import h5py
import oyaml as yaml
import torch

from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.flmts_reactors_data_gen.main import generate_flmts_database
from h2_learning_database import write_h2_learning_database

import DummyComm as dc

mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

# Fuel inlet temperature [K], below the air inlet temperatures
T_fuel = 350.0

data_gen_parameters = {
    "mech_file": mech_file,
    "pv_species": ["H2O"],
    "fuel": ["H2"],
    "results_folder_suffix": "DIFF_TEST",
    "dtb_file": "solutions.h5",
    "pressure": 101325.0,
    "time_step": 1.0e-6,
    "include_zerod": False,
    "include_oned_prem": False,
    "include_oned_diff": True,
    "oned_diff_params": {"strain_bounds": [100.0, 200.0], "T0_bounds": [600.0, 700.0], "n_samples": 2, "width": 0.02,
                         "T_fuel": T_fuel},
    "augment_dataset": False,
    "scatter_plot_variables": ["Temperature", "OH"],
}


def test_flamelets_diffusion_settings(tmp_path):
    os.chdir(tmp_path)
    with open("dtb_params_flmts.yaml", "w") as file:
        yaml.dump(data_gen_parameters, file, default_flow_style=False)

    generate_flmts_database(dc.DummyComm())

    folder = tmp_path / "FLAMELETS_DTB_DIFF_TEST"
    with h5py.File(folder / "solutions.h5", "r") as f:
        T = f["ITERATION_00000/X"][:, 0]
    assert abs(T.min() - T_fuel) < 1.0e-6

    assert os.path.isfile(folder / "scatter_plot_Temperature_OH.png")


def test_plateau_scheduler_settings(tmp_path, monkeypatch):
    os.chdir(tmp_path)
    write_h2_learning_database(tmp_path)

    with open("networks_params.yaml") as file:
        params = yaml.safe_load(file)
    params["learning"].update({"scheduler_option": "ReduceLROnPlateau", "plateau_factor": 0.3, "plateau_patience": 4})
    with open("networks_params.yaml", "w") as file:
        yaml.dump(params, file, default_flow_style=False)

    # Arguments of the scheduler built by train_all_clusters
    built = {}
    scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau

    def scheduler(optimizer, **kwargs):
        built.update(kwargs)
        return scheduler_class(optimizer, **kwargs)

    monkeypatch.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", scheduler)

    NN_manager().train_all_clusters()

    assert built["factor"] == 0.3 and built["patience"] == 4
