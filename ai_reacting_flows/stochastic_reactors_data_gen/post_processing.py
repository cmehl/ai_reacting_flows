import os

import numpy as np
import pandas as pd
import cantera as ct
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

import ai_reacting_flows.tools.utilities as utils


class StochDatabase(object):
    
    def __init__(self, stoch_dtb_folder, save_folder):
        
        self.stoch_dtb_folder = stoch_dtb_folder

        # By default, no 
        self.add_0D_ignition_archetype = False
        self.add_1D_premixed_archetype = False

        # Loading database
        dtb = stoch_dtb_folder + "/database_states.csv"
        self.df = pd.read_csv(dtb, sep=';')

        # Loading trajectories
        traj_data = stoch_dtb_folder + "/mean_trajectories.h5"
        f = h5py.File(traj_data,"r")
        #
        traj_dataset = f['TRAJECTORIES']
        nb_inlets = 0 # number of inlets
        self.inlets_data_list = {}  # dictionary for storing data
        for inlet in traj_dataset.keys():
            i = int(inlet[-1])
            self.inlets_data_list[i] = np.asarray(traj_dataset[inlet])
            nb_inlets += 1
        #
        # Getting number of states variable
        self.nb_state_vars = self.inlets_data_list[1].shape[1] - 3  # all except Z, phi and time
        # 
        f.close()

        # Saving folder
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        self.save_folder = save_folder

    #--------------------------------------------------------
    # CANONICAL FLAMES CALCULATION
    #--------------------------------------------------------

    def compute_0D_ignition(self, phi, p, T0, fuel, mech_file):

        self.add_0D_ignition_archetype = True
        
        # Compute flame
        self.T_cano_0D, self.Y_cano_dict_0D = utils.compute_0D_reactor(fuel, mech_file, phi, T0, p)


    def compute_1D_premixed(self, phi, p, T0, fuel, mech_file, diffusion_model):

        self.add_1D_premixed_archetype = True

        # Conditions of flame
        phi = 0.4
        p = 101325.0
        T0 = 300.0

        # Compute flame
        self.T_cano_1D, self.Y_cano_dict_1D = utils.compute_adiabatic(fuel, mech_file, phi, T0, p, diffusion_model)


    #--------------------------------------------------------
    # SCATTER PLOTS
    #--------------------------------------------------------

    def plot_T_Z(self):

        # Creating axis
        fig, ax = plt.subplots()

        self.df.plot.scatter(x='Mix_frac', y='Temperature', ax=ax, c='Time', colormap='viridis')
        ax.set_xlabel(r"$Z$ $[-]$")
        ax.set_ylabel(r"$T$ $[K]$")

        ax.set_xlim([0.9*self.df['Mix_frac'].min(), 1.1*self.df['Mix_frac'].max()])

        fig.tight_layout()

        # Save
        fig.savefig(self.save_folder + "/dtb_TZ_plot.png", dpi=300)


    def plot_Z_Yk(self, species_to_plot):

        for spec in species_to_plot:
            
            fig, ax = plt.subplots()
            
            self.df.plot.scatter(x='Mix_frac', y=spec, ax=ax, c='Time', colormap='viridis')
            ax.set_xlabel(r"$Z$ $[-]$")
            ax.set_ylabel(f"${spec}$ mass fraction $[-]$")
            
            ax.set_xlim([0.9*self.df['Mix_frac'].min(), 1.1*self.df['Mix_frac'].max()])
            
            fig.tight_layout()
            
            # Save
            fig.savefig(self.save_folder + f"/dtb_{spec}_Z_plot.png", dpi=300)


    def plot_T_Yk(self, species_to_plot):

        # Creating axis
        for spec in species_to_plot:
            
            fig, ax = plt.subplots()
            
            self.df.plot.scatter(x='Temperature', y=spec, ax=ax, c='Time', colormap='viridis')
            
            # Canonical flame structures
            if self.add_1D_premixed_archetype:
                ax.plot(self.T_cano, self.Y_cano_dict[spec], color='r', lw=3, ls='--', label="Laminar")
                
            if self.add_0D_ignition_archetype:
                ax.plot(self.T_cano_0D, self.Y_cano_dict_0D[spec], color='b', lw=3, ls='--', label="Ignition")
            
            ax.set_xlabel(r"$T$ $[K]$")
            ax.set_ylabel(f"${spec}$ mass fraction $[-]$")
            ax.legend()
            
            ax.set_xlim([0.9*self.df['Temperature'].min(), 1.1*self.df['Temperature'].max()])
            
            fig.tight_layout()
            
            plt.show()

            # Save
            fig.savefig(self.save_folder + f"/dtb_T_{spec}_plot.png", dpi=300)



    #--------------------------------------------------------
    # TRAJECTORIES PLOTS
    #--------------------------------------------------------

    def plot_traj_T_Z(self):

        # styles
        linestyles = ["--", "-", "-.", ":"]

        fig, ax = plt.subplots()

        j = 0   
        for i in self.inlets_data_list.keys():
            ax.plot(self.inlets_data_list[i][:,self.nb_state_vars+1], self.inlets_data_list[i][:, 1], color="k", linestyle=linestyles[j], lw=2, label=f"Inlet {i:d}")
            j += 1    
            
        ax.set_xlabel(r"$Z$ $[-]$")
        ax.set_ylabel(r"$T$ $[K]$")

        ax.set_xlim([0.9*np.min(self.inlets_data_list[i][:,self.nb_state_vars+1]), 1.1*np.max(self.inlets_data_list[i][:,self.nb_state_vars+1])])

        ax.legend()

        fig.tight_layout()

        # Save
        fig.savefig(self.save_folder + "traj_TZ_plot.png", dpi=300)


    def plot_traj_T_time(self):

        # styles
        linestyles = ["--", "-", "-.", ":"]

        fig, ax = plt.subplots()

        j = 0   
        for i in self.inlets_data_list.keys():
            ax.plot(self.inlets_data_list[i][:,0], self.inlets_data_list[i][:, 1], color="k", linestyle=linestyles[j], lw=2, label=f"Inlet {i:d}")
            j += 1    
            
        ax.set_xlabel(r"$t$ $[s]$")
        ax.set_ylabel(r"$T$ $[K]$")

        fig.tight_layout()

        ax.legend()


    def plot_traj_Yk_time(self, species_to_plot, mech_file):

        # To get species index, maybe to be put in __init__ at some point
        gas = ct.Solution(mech_file)

        for spec in species_to_plot:
    
            fig, ax = plt.subplots()
            
            j = 0   
            for i in self.inlets_data_list.keys():
                ax.plot(self.inlets_data_list[i][:,0], self.inlets_data_list[i][:, 3+gas.species_index(spec)], color="k", lw=2, linestyle=self.linestyles[j], label=f"Inlet {i:d}")
                j += 1
                
            ax.set_xlabel(r"$t$ $[s]$")
            ax.set_ylabel(f"${spec}$ mass fraction $[-]$")
            
            ax.legend()
            
            fig.tight_layout()
            
            # Save
            fig.savefig(self.save_folder + f"traj_{spec}_time_plot.png", dpi=300)


    #--------------------------------------------------------
    # INDIVIDUAL PARTICLES TRACKING
    #--------------------------------------------------------

    def plot_indiv_traj(self, inlet_nb, var):

        # Filter only desired inlet 
        inlet_df = self.df[self.df["Inlet_number"]==inlet_nb]

        # Get list of dataframes for each particle
        df_part_list = list(inlet_df.groupby('Particle_number'))

        # Create figure
        fig, ax = plt.subplots()

        for item in df_part_list:
            
            df_part = item[1]
            
            ax.plot(df_part["Time"], df_part[var], alpha=0.2, color="purple")

        fig.tight_layout()

        # Save
        fig.savefig(self.save_folder + f"/indiv_rajectory.png", dpi=300)

























