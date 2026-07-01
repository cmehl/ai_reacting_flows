import os
import oyaml as yaml

from time import perf_counter

import numpy as np
import h5py
import pandas as pd
import cantera as ct

from mpi4py import MPI

h5py.get_config().mpi

def GenerateVariable_dt(dtb_type, params, comm : 'MPI.Comm'):

    rank = comm.Get_rank()
    size = comm.Get_size()

    np.random.seed(1234)

    if dtb_type=="stoch":
        dtb_prefix = "STOCH"
    elif dtb_type=="flamelets":
        dtb_prefix = "FLAMELETS"

    # Opening h5 file
    run_folder = os.getcwd()
    stoch_results_folder = f"{run_folder:s}/{dtb_prefix}_DTB_" + params["results_folder_suffix"]
    file_h5 = f"{stoch_results_folder:s}/{params['dtb_file'].split('.')[0]:s}.h5"

    if dtb_type=="stoch":
        X, col_names_X, col_names_Y= read_database_stoch(file_h5)
    elif dtb_type=="flamelets":
        X, col_names_X, col_names_Y = read_database_flmts(file_h5)

    if rank==0: print(f"X columns: {col_names_X} \n")

    Y = np.empty(X.shape)
    # np.random.shuffle(X) # DAK: check why we shuffle X
    state_list = np.dstack((X,Y)) # DAK : check why we dstack with Y (np.empty)

    #CM: that doubles the threshold already applied in database processing and might lead to confusion
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

        # For multi-dt stochastic all data is written by default in "ITERATION_00000" as it is too complex to keep the per iteration structure 
        # (and useless as it is not considered when creating processed database)
        if dtb_type=="stoch":
            group_name = "ITERATION_00000"
        elif dtb_type=="flamelets":
            group_name = "FLAMELETS"
        #
        print('WRITING', rank)
        f = h5py.File(f"{stoch_results_folder:s}/{params['new_file_name']:s}", "w")
        grp = f.create_group(group_name)
        dset_X = grp.create_dataset('X', data=X_new_full)
        dset_Y = grp.create_dataset('Y', data=Y_new_full)
        dset_X.attrs["cols"] = np.append(col_names_X, 'dt')
        dset_Y.attrs["cols"] = col_names_Y
        f.close()
        print(f'Processor {rank} has written output file. \n')
            
    comm.Barrier()
    
    return 'Done'


#TODO : mutualize these 2 functions with other scripts (e.g. data processing) by adding them to utilities

def read_database_stoch(file_h5):

    h5file_r = h5py.File(file_h5, 'r')

    col_names_X = h5file_r[f"ITERATION_00000/X"].attrs["cols"] #[:-2]    # -2 because we remove c and HRR which are in those arrays in the h5 file
    col_names_Y = h5file_r[f"ITERATION_00000/Y"].attrs["cols"] #[:-2]

    # Loop on solutions
    list_df_X = []

    nb_solutions = len(h5file_r.keys())

    for i in range(nb_solutions):
        data_X = h5file_r.get(f"ITERATION_{i:05d}/X") #[:,:-2]
        list_df_X.append(pd.DataFrame(data=data_X, columns=col_names_X))

    h5file_r.close()

    X = pd.concat(list_df_X, ignore_index=True).to_numpy().copy()

    return X, col_names_X, col_names_Y


def read_database_flmts(file_h5):

    h5file_r = h5py.File(file_h5, 'r')

    col_names_X = h5file_r[f"FLAMELETS/X"].attrs["cols"] #[:-2]    # -2 because we remove c and HRR which are in those arrays in the h5 file
    col_names_Y = h5file_r[f"FLAMELETS/Y"].attrs["cols"] #[:-2]

    data_X = h5file_r.get(f"FLAMELETS/X") #[:,:-2]
    X = pd.DataFrame(data=data_X, columns=col_names_X)

    h5file_r.close()

    return X, col_names_X, col_names_Y


def react_multi_dt(state, gas, T_thresh, dt_list):

    # Unsolved pb: if Yk from NN is inputed here; 
    # it may become negative and mass is lost (output from CVODE is always positive)
    time_step = np.array([[0, 0]])
    state = np.append(state, time_step, axis = 0)
    new_states = np.empty_like([state,])

    if state[0,0] > T_thresh:  

        # Initial value are current's particle state
        T0 = state[0,0]
        P0 = state[1,0]
        Y0 = state[2:-3,0]

        # Advancing to dts
        for dt in dt_list:
            
            gas.TPY = T0, P0, Y0
                        
            # Constant pressure reactor
            r = ct.IdealGasConstPressureReactor(gas)
            # Initializing reactor
            sim = ct.ReactorNet([r])

            # Advancing simulation by dt
            state[-1, 0] = dt
            sim.advance(dt)

            # Updated state
            y = np.empty(len(Y0)+4)  # T, p, c, HRR
            y[0] = gas.T
            y[1] = gas.P
            y[2:-2] = gas.Y
            y[-2] = -1.0  # dummy vlaue for progvar
            y[-1] = -1.0  # dummy value for HRR

            state[:,1] = np.append(y, [0])
            new_states = np.append(new_states, np.array([state,]), axis = 0)
        
        new_states = new_states[1:,]  # First was empty and only used for using append in loop

        return new_states
    
    return None