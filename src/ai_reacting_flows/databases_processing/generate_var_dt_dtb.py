import os
import oyaml as yaml

from time import perf_counter

from ai_reacting_flows.tools.utilities import PRINT, react_multi_dt

import numpy as np
import h5py
import pandas as pd
import cantera as ct

from mpi4py import MPI

h5py.get_config().mpi

def GenerateVariable_dt(params, comm : 'MPI.Comm'):
    rank = comm.Get_rank()
    size = comm.Get_size()

    np.random.seed(1234)

    # Opening h5 file
    run_folder = os.getcwd()
    stoch_results_folder = f"{run_folder:s}/STOCH_DTB_" + params["results_folder_suffix"]
    # h5file_r = h5py.File(f"{stoch_results_folder:s}/{params['dtb_file'].split('.')[0]:s}_resamp{int(params['T_threshold'])}K.h5", 'r')
    h5file_r = h5py.File(f"{stoch_results_folder:s}/{params['dtb_file'].split('.')[0]:s}.h5", 'r')

    # Solution 0 read to get columns names
    col_names_X = h5file_r["ITERATION_00000/X"].attrs["cols"][:-2]    # -2 because we remove c and HRR which are in those arrays in the h5 file
    col_names_Y = h5file_r["ITERATION_00000/Y"].attrs["cols"][:-2]

    print(f"X columns: {col_names_X} \n")

    # Loop on solutions
    list_df_X = []

    nb_solutions = len(h5file_r.keys())

    for i in range(nb_solutions):
        data_X = h5file_r.get(f"ITERATION_{i:05d}/X")[:,:-2]
        list_df_X.append(pd.DataFrame(data=data_X, columns=col_names_X))

    h5file_r.close()

    X = pd.concat(list_df_X, ignore_index=True).to_numpy().copy()
    Y = np.empty(X.shape)
    np.random.shuffle(X) # DAK: check why we shuffle X
    state_list = np.dstack((X,Y)) # DAK : check why we dstack with Y (np.empty)

    # Temperature threshold
    T_thresh = params['T_threshold']

    # Adding dt to the features
    dt_simu = params['time_step']
    if params['time_step_type'] == 'set':
        dt_list = params['time_step_range']
        if dt_simu not in dt_list:
            dt_list.append(dt_simu)
    else:
        dt_range = params['time_step_range']
        nb_dt = params['nb_dt']
        dt_min, dt_max = dt_range[0], dt_range[1]
        fact_min, fact_max = 1.0, 1.0  # Extend the learned dt 

    state_list = np.array_split(state_list, size)
    # assert len(state_list) == size

    state_list = comm.scatter(state_list, root=0)
    
    gas = ct.Solution(params["mech_file"])
    # print('appel_gas')
    tot_0 = len(state_list)

    count = 0
    all_new_states = []
    # Reacting all states X for dt in dt_range
    for state in state_list: 
        count +=1
        if rank == 0 and count%5000 == 0:
            print(f"Operation (proc 0) : {count} / {int(tot_0)}")
        if params['time_step_type'] == 'random':
            #Sample dt
            dt_list = np.random.uniform(low=np.log(fact_min*dt_min), high=np.log(fact_max*dt_max), size=(nb_dt))
            dt_list = np.exp(dt_list)
            # dt_list = sc.stats.loguniform(dt_range[0], dt_range[1]).rvs(size = nb_dt) 
            dt_list.sort()  # CM: is it necessary ?
        new_states = react_multi_dt(state, gas, T_thresh, dt_list)
        if new_states is not None:
            all_new_states.append(new_states)
    
    
    all_new_states = np.concatenate(all_new_states)
    X_new, Y_new = np.dsplit(all_new_states, 2)
    X_new, Y_new = np.squeeze(X_new), np.squeeze(Y_new)
    Y_new = Y_new[:,:-1]

    print(f'Processor {rank} has built array all_new_states. \n')
    
    comm.Barrier()

    # Gather all X_new/Y_new arrays from every rank onto rank 0
    all_X_new = comm.gather(X_new, root=0)
    all_Y_new = comm.gather(Y_new, root=0)

    if rank ==0:

        # Concatenate the per-rank arrays into one global array
        X_new_full = np.concatenate(all_X_new, axis=0)
        Y_new_full = np.concatenate(all_Y_new, axis=0)

        # For multi-dt all data is written by default in "ITERATION_00000" as it is too complex to keep the per iteration structure 
        # (and useless as it is not considered when creating processed database)
        print('WRITING', rank)
        f = h5py.File(f"{stoch_results_folder:s}/{params['new_file_name']:s}", "w")
        grp = f.create_group("ITERATION_00000")
        dset_X = grp.create_dataset('X', data=X_new_full)
        dset_Y = grp.create_dataset('Y', data=Y_new_full)
        dset_X.attrs["cols"] = np.append(col_names_X, 'dt')
        dset_Y.attrs["cols"] = col_names_Y
        f.close()
        print(f'Processor {rank} has written output file. \n')
            
    comm.Barrier()
    
    return 'Done'