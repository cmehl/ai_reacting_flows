from scipy.stats import wasserstein_distance

import h5py
import numpy as np
import pandas as pd

def extract_h2_dtb_histograms(dtb_file, histo_folder):
    h5file_r = h5py.File(dtb_file, 'r')
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
        # np.savez(f"H2_dtvar_TEST_histograms/{var:s}_hist.npz", counts=counts, bin_edges=bin_edges)

        data = np.load(f"{histo_folder:s}/{var:s}_hist.npz")
        counts_ref = data["counts"]
        bin_edges_ref = data["bin_edges"]

        #threshold found using the following code
        # df["noisy"] = df[var] * (1.0 + np.random.normal(loc=0,scale=0.01,size=len(df)))
        # counts, _ = np.histogram(df["noisy"], bins=bin_edges_ref)
        # print(f"distance for {var:s} is {wasserstein_distance(counts, counts_ref):f}")
        
        # if var in ["Temperature","H2","O2"]:
        # print(f"Histograms for {var:s}:")
        # print(counts)
        # print(counts_ref)
        # print(bin_edges_ref)
        
        counts, _ = np.histogram(df[var], bins=bin_edges_ref)
        distances.append(wasserstein_distance(counts, counts_ref))
        print(f"distance for {var:s} is {distances[-1]:f}")
    
    res = np.array(distances)
    # print(res)
    res -= np.array([139.75, 3.74, 50.4, 4.94, 166.15, 9.9, 124.38, 5.9, 4.62, 2.04])
    return res

def extract_h2_dtvar_dtb_histograms(dtb_file, histo_folder):
    h5file_r = h5py.File(dtb_file, 'r')
    names = h5file_r.keys()
    nb_solutions = len(names)
    print(nb_solutions)
    
    col_names = h5file_r["ITERATION_00000/Y"].attrs["cols"]
    data = h5file_r.get("ITERATION_00000/Y")[()]
    df = pd.DataFrame(data=data, columns=col_names)
        
    varlist = col_names.tolist()
    varlist = [var for var in varlist if var not in ["Pressure", "N2"]]
    distances = []
    for var in varlist:
        # references histogram generated using the following code (to use again if reference is to change)
        # counts, bin_edges = np.histogram(df[var], bins=100)
        # bin_edges[-1] *= np.float64(1.01)
        # if var not in ["Temperature"]:
        #     bin_edges[-1] = np.min([bin_edges[-1], 1.0], axis=0)
        # if var != "HRR":
        #     bin_edges[0] *= np.float64(0.99)
        # else:
        #     bin_edges[0] = bin_edges[0] * np.float64(0.99) if bin_edges[0] > 0.0 else bin_edges[0] * np.float64(1.01)
        # counts, _ = np.histogram(df[var], bins=bin_edges)
        # np.savez(f"{histo_folder:s}/{var:s}_hist.npz", counts=counts, bin_edges=bin_edges)

        data = np.load(f"{histo_folder:s}/{var:s}_hist.npz")
        counts_ref = data["counts"]
        bin_edges_ref = data["bin_edges"]

        #threshold found using the following code
        # df["noisy"] = df[var] * (1.0 + np.random.normal(loc=0,scale=0.01,size=len(df)))
        # counts, _ = np.histogram(df["noisy"], bins=bin_edges_ref)
        # print(f"distance for {var:s} is {wasserstein_distance(counts, counts_ref):f}")

        # if var in ["Temperature","H2","O2"]:
        # print(f"Histograms for {var:s}:")
        # print(counts)
        # print(counts_ref)
        # print(bin_edges_ref)

        counts, _ = np.histogram(df[var], bins=bin_edges_ref)
        distances.append(wasserstein_distance(counts, counts_ref))
        # print(f"distance for {var:s} is {distances[-1]:f}")
    
    res = np.array(distances)
    # print(res)
    res -= np.array([685.15, 11.62, 253.73, 18.62, 825.96, 43.68, 610.84, 12.53, 11.26])
    return res