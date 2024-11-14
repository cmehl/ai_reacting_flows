from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database
from scipy.stats import wasserstein_distance

import DummyComm as dc
import h5py
import numpy as np
import pandas as pd

def test_h2_dtb_computation():
    # Dictionary to store inputs
    data_gen_parameters = {}

    # General parameters
    data_gen_parameters["mech_file"] = "../data/chemical_mechanisms/mech_H2.yaml"      # Mechanism file
    data_gen_parameters["inlets_file"] = "inlets_file.xlsx"      # file defining the stochastic reactors inlets
    data_gen_parameters["results_folder_suffix"] = "TEST"    # Name of the resulting database (used as a suffix of "STOCH_DTB")
    data_gen_parameters["build_ml_dtb"] = True     # Flag to generate ML database or do a stand-alone simulation
    data_gen_parameters["time_step"] = 5.0e-6       # Constant time step of the simulation
    data_gen_parameters["time_max"] = 1e-3          # Simulation end time

    # Modeling parameters
    data_gen_parameters["mixing_model"] = "CURL_MODIFIED"         # Mixing model (CURL or CURL_MODIFIED)
    data_gen_parameters["mixing_time"] = 1.0e-4                 # Mixing time-scale
    data_gen_parameters["read_mixing"] = True                 # Use pre-computed mixing pairs
    data_gen_parameters["T_threshold"] = 200.0                  # Temperature threshold for database
    data_gen_parameters["calc_progvar"] = False                  # Flag to decide if progress variable is computed
    data_gen_parameters["pv_species"] = ["CO", "CO2"]                 # Species used to define progress variable

    # post-processing parameters
    data_gen_parameters["calc_mean_traj"] = False              # Flag to decide if mean trajectories are computed in the post-processing step

    # ML database parameters
    data_gen_parameters["dt_ML"] = 0.5e-6                      # Time step of the ML database (may be different from the simulation time step)

    # ML inference
    data_gen_parameters["ML_inference_flag"] = False
    data_gen_parameters["ML_models"] = ("")
    data_gen_parameters["prog_var_thresholds"] = (0.25, 0.95)

    # Call to database generation function
    generate_stochastic_database(data_gen_parameters, dc.DummyComm())

    res = extract_h2_dtb_histograms()
    print(res)
    assert np.max(res) < np.float64(0.0)

def extract_h2_dtb_histograms():
    h5file_r = h5py.File("./STOCH_DTB_TEST/solutions.h5", 'r')
    names = h5file_r.keys()
    nb_solutions = len(names)
    print(nb_solutions)
    
    col_names = h5file_r["ITERATION_00000/all_states"].attrs["cols"]
    list_df = []
    for i in range(nb_solutions):

        if i%100==0:
            print(f"Opening solution: {i} / {nb_solutions}")

        data = h5file_r.get(f"ITERATION_{i:05d}/all_states")[()]

        list_df.append(pd.DataFrame(data=data, columns=col_names))
    
    df = pd.concat(list_df, ignore_index=True)
    
    varlist = col_names.tolist()
    varlist = [var for var in varlist if var not in ["Prog_var","N2","Pressure","Time","mass","Y_C","Y_H","Y_O","Y_N","Inlet_number","Particle_number","Mix_frac", "Equiv_ratio", "Mass"]]
    distances = []
    for var in varlist:
        # references histogram generated using the following code (to use again if reference is to change)
        # counts, bin_edges = np.histogram(df[var], bins=100)
        # bin_edges[-1] *= np.float64(1.01)
        # if var not in ["Temperature", "HRR"]:
        #     bin_edges[-1] = np.min([bin_edges[-1], 1.0], axis=0)
        # if var != "HRR":
        #     bin_edges[0] *= np.float64(0.99)
        # else:
        #     bin_edges[0] = bin_edges[0] * np.float64(0.99) if bin_edges[0] > 0.0 else bin_edges[0] * np.float64(1.01)
        # counts, _ = np.histogram(df[var], bins=bin_edges)
        # np.savez(f"H2_TEST_histograms/{var:s}_hist.npz", counts=counts, bin_edges=bin_edges)

        data = np.load(f"H2_TEST_histograms/{var:s}_hist.npz")
        counts_ref = data["counts"]
        bin_edges_ref = data["bin_edges"]

        #threshold found using the following code
        # df["noisy"] = df[var] * (1.0 + np.random.normal(loc=0,scale=0.01,size=len(df)))
        # counts, _ = np.histogram(df["noisy"], bins=bin_edges_ref)
        
        # if var in ["Temperature","H2","O2"]:
        #     print(f"Histograms for {var:s}:")
        #     print(counts)
        #     print(counts_ref)
        #     print(bin_edges_ref)
        
        counts, _ = np.histogram(df[var], bins=bin_edges_ref)
        distances.append(wasserstein_distance(counts, counts_ref))
        # print(f"distance for {var:s} is {wasserstein_distance(counts, counts_ref):f}")
    
    res = np.array(distances)
    # print(res)
    res -= np.array([139.75, 3.74, 50.4, 4.94, 166.15, 9.9, 124.38, 5.9, 4.62, 2.04])
    return res