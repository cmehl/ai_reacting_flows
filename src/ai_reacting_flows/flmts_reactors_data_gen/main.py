import os
import shutil
import oyaml as yaml

from mpi4py import MPI

from ai_reacting_flows.flmts_reactors_data_gen.database_flmts import DatabaseFlamelets
from ai_reacting_flows.tools.utilities import PRINT


def generate_flmts_database(comm : 'MPI.Comm'):

    run_folder = os.getcwd()
    with open(os.path.join(run_folder, "dtb_params_flmts.yaml"), "r") as file:
        data_gen_parameters = yaml.safe_load(file)

    rank = comm.Get_rank()

    #==========================================================================
    # INITIALIZING RUN
    #==========================================================================

    if rank==0:

        PRINT("")
        PRINT(60 * "=")
        PRINT(" GENERATION OF FLAMELETS DATABASE ")
        PRINT(60 * "=")

        # Remove existing results dir if already exists
        results_folder = "FLAMELETS_DTB_" + data_gen_parameters["results_folder_suffix"]
        if os.path.isdir(results_folder):
            PRINT("WARNING: existing results directory has been removed. \n")
            shutil.rmtree(results_folder)
        os.mkdir(results_folder)

        # Copying input files to results folder to document run
        shutil.copy(data_gen_parameters["mech_file"], results_folder)


    # Flamelet database generation
    dtb_flmts = DatabaseFlamelets(data_gen_parameters, comm)

    #==========================================================================
    # BUILDING DATABASE
    #==========================================================================

    # Computing flamelets of different types
    # 0D reactors
    if data_gen_parameters["include_zerod"]:
        if rank==0:
            PRINT(" >>> Computing 0D reactors")
        zerod_params = data_gen_parameters["zerod_params"]
        dtb_flmts.compute_0d_reactors(zerod_params)
        if rank==0:
            PRINT("")

    # 1D premixed flames
    if data_gen_parameters["include_oned_prem"]:
        if rank==0:
         PRINT(" >>> Computing 1D premixed flames")
        oned_prem_params = data_gen_parameters["oned_prem_params"]
        dtb_flmts.compute_1d_premixed(oned_prem_params)
        if rank==0:
            PRINT("")

    # 1D diffusion flames
    if data_gen_parameters["include_oned_diff"]:
        if rank==0:
            PRINT(" >>> Computing 1D diffusion flames")
        oned_diff_params = data_gen_parameters["oned_diff_params"]
        dtb_flmts.compute_1d_diffusion(oned_diff_params)
        if rank==0:
            PRINT("")

    # Augment dataset with HFRD method
    if data_gen_parameters["augment_dataset"]:
        if rank==0:
            PRINT(" >>> Data augmentation")
        dtb_flmts.augment_data()
        if rank==0:
            PRINT("")

    # Saving database
    if rank==0:
        dtb_flmts.save_database()

    # Scatter plot to visualize database
    if rank==0:
        x_var = "Temperature"
        y_var = "H2O"   # CM: add maybe as input in yaml file
        dtb_flmts.scatter_plot_data(x_var, y_var)

    # generate the X, Y database (untransformed data)
    dtb_flmts.generate_XY_h5()

    #==============================================================================
    # ENDING THE SIMULATION
    #==============================================================================

    if rank==0:

        shutil.copy(os.path.join(run_folder, "dtb_params_flmts.yaml"), f"{run_folder:s}/FLAMELETS_DTB_{data_gen_parameters['results_folder_suffix']}/dtb_params_flmts.yaml")

        PRINT("")
        PRINT("END OF SIMULATION")
        PRINT("")

    return