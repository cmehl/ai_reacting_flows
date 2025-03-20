import os

import numpy as np
import pandas as pd
from scipy.interpolate import interpn
import cantera as ct
import h5py

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize 
from matplotlib import cm

import seaborn as sns

import ai_reacting_flows.tools.utilities as utils

sns.set_style("darkgrid")

class StochDatabase(object):
    
    def __init__(self, stoch_dtb_folder, save_folder):
        
        self.stoch_dtb_folder = stoch_dtb_folder

        # By default, no 
        self.add_0D_ignition_archetype = False
        self.add_1D_premixed_archetype = False

        # Loading database: concatenation of each data in h5 file
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')
        names = h5file_r.keys()
        self.nb_solutions = len(names)
        h5file_r.close()
        self.get_all_states()

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

        # Additional post_processing
        self.compute_additional_postpros()

        # Saving folder
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        self.save_folder = save_folder

    #--------------------------------------------------------
    # READING H5 SOLUTION FILES
    #--------------------------------------------------------

    def get_all_states(self):

        # Reading file
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')

        # Solution 0 read to get columns names
        col_names = h5file_r["ITERATION_00000/all_states"].attrs["cols"]

        # Loop on solutions
        list_df = []
        for i in range(self.nb_solutions):

            if i%100==0:
                print(f"Opening solution: {i} / {self.nb_solutions}")

            data = h5file_r.get(f"ITERATION_{i:05d}/all_states")[()]

            list_df.append(pd.DataFrame(data=data, columns=col_names))
        
        self.df = pd.concat(list_df, ignore_index=True)

        h5file_r.close()
        
    #--------------------------------------------------------
    # CANONICAL FLAMES CALCULATION
    #--------------------------------------------------------

    def compute_0D_ignition(self, phi, p, T0, fuel, mech_file):

        self.add_0D_ignition_archetype = True
        
        # Compute flame
        self.T_cano_0D, self.Y_cano_dict_0D = utils.compute_0D_reactor(fuel, mech_file, phi, T0, p)

    def compute_1D_premixed(self, phi, p, T0, fuel, mech_file, diffusion_model):

        self.add_1D_premixed_archetype = True

        # Compute flame
        self.T_cano_1D, self.Y_cano_dict_1D = utils.compute_adiabatic(fuel, mech_file, phi, T0, p, diffusion_model)

    #--------------------------------------------------------
    # ADDITIONAL CALCULATIONS
    #--------------------------------------------------------

    def compute_additional_postpros(self):

        self.df["abs_HRR"] = np.abs(self.df["HRR"])
        self.df["log_abs_HRR"] = np.log(self.df["abs_HRR"])

    #--------------------------------------------------------
    # SCATTER PLOTS: ALL STATES
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
                ax.plot(self.T_cano_1D, self.Y_cano_dict_1D[spec], color='r', lw=3, ls='--', label="Laminar")
                
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

    # Generic plotting function

    def plot_generic(self, var_x, var_y, var_c):

        # Creating axis
        fig, ax = plt.subplots()

        self.df.plot.scatter(x=var_x, y=var_y, ax=ax, c=var_c, colormap='viridis')
        ax.set_xlabel(var_x)
        ax.set_ylabel(var_y)

        ax.set_xlim([0.9*self.df[var_x].min(), 1.1*self.df[var_x].max()])

        fig.tight_layout()

        # Save
        fig.savefig(self.save_folder + f"/dtb_x{var_x}_y{var_y}_c{var_c}_plot.png", dpi=300)

    #--------------------------------------------------------
    # SCATTER PLOTS: ONE SOLUTION
    #--------------------------------------------------------

    def plot_T_Z_indiv(self, iteration):

        # Loading solution at given iteration
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')
        data = h5file_r.get(f"ITERATION_{iteration:05d}/all_states")[()]
        col_names = h5file_r[f"ITERATION_{iteration:05d}/all_states"].attrs["cols"]
        h5file_r.close()
        df = pd.DataFrame(data=data, columns=col_names)

        # Creating axis
        fig, ax = plt.subplots()

        df.plot.scatter(x='Mix_frac', y='Temperature', ax=ax, c='Time', colormap='viridis')
        ax.set_xlabel(r"$Z$ $[-]$")
        ax.set_ylabel(r"$T$ $[K]$")

        ax.set_xlim([0.9*df['Mix_frac'].min(), 1.1*df['Mix_frac'].max()])

        fig.tight_layout()

        # Save
        fig.savefig(self.save_folder + f"/dtb_TZ_plot_iteration{iteration:05d}.png", dpi=300)

    def plot_Z_Yk_indiv(self, species_to_plot, iteration):

        # Loading solution at given iteration
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')
        data = h5file_r.get(f"ITERATION_{iteration:05d}/all_states")[()]
        col_names = h5file_r[f"ITERATION_{iteration:05d}/all_states"].attrs["cols"]
        h5file_r.close()
        df = pd.DataFrame(data=data, columns=col_names)

        for spec in species_to_plot:
            
            fig, ax = plt.subplots()
            
            df.plot.scatter(x='Mix_frac', y=spec, ax=ax, c='Time', colormap='viridis')
            ax.set_xlabel(r"$Z$ $[-]$")
            ax.set_ylabel(f"${spec}$ mass fraction $[-]$")
            
            ax.set_xlim([0.9*df['Mix_frac'].min(), 1.1*df['Mix_frac'].max()])
            
            fig.tight_layout()
            
            # Save
            fig.savefig(self.save_folder + f"/dtb_{spec}_Z_plot_iteration{iteration:05d}.png", dpi=300)

    def plot_T_Yk_indiv(self, species_to_plot, iteration):

        # Loading solution at given iteration
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')
        data = h5file_r.get(f"ITERATION_{iteration:05d}/all_states")[()]
        col_names = h5file_r[f"ITERATION_{iteration:05d}/all_states"].attrs["cols"]
        h5file_r.close()
        df = pd.DataFrame(data=data, columns=col_names)

        # Creating axis
        for spec in species_to_plot:
            
            fig, ax = plt.subplots()
            
            df.plot.scatter(x='Temperature', y=spec, ax=ax, c='Time', colormap='viridis')
            
            # Canonical flame structures
            if self.add_1D_premixed_archetype:
                ax.plot(self.T_cano, self.Y_cano_dict[spec], color='r', lw=3, ls='--', label="Laminar")
                
            if self.add_0D_ignition_archetype:
                ax.plot(self.T_cano_0D, self.Y_cano_dict_0D[spec], color='b', lw=3, ls='--', label="Ignition")
            
            ax.set_xlabel(r"$T$ $[K]$")
            ax.set_ylabel(f"${spec}$ mass fraction $[-]$")
            ax.legend()
            
            ax.set_xlim([0.9*df['Temperature'].min(), 1.1*df['Temperature'].max()])
            
            fig.tight_layout()
            
            plt.show()

            # Save
            fig.savefig(self.save_folder + f"/dtb_T_{spec}_plot_{iteration:05d}.png", dpi=300)

    # Generic plotting function

    def plot_generic_indiv(self, var_x, var_y, var_c, iteration):

        # Loading solution at given iteration
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')
        data = h5file_r.get(f"ITERATION_{iteration:05d}/all_states")[()]
        col_names = h5file_r[f"ITERATION_{iteration:05d}/all_states"].attrs["cols"]
        h5file_r.close()
        df = pd.DataFrame(data=data, columns=col_names)

        # Creating axis
        fig, ax = plt.subplots()

        df.plot.scatter(x=var_x, y=var_y, ax=ax, c=var_c, colormap='viridis')
        ax.set_xlabel(var_x)
        ax.set_ylabel(var_y)

        ax.set_xlim([0.9*df[var_x].min(), 1.1*df[var_x].max()])

        fig.tight_layout()

        # Save
        fig.savefig(self.save_folder + f"/dtb_x{var_x}_y{var_y}_c{var_c}_plot_{iteration:05d}.png", dpi=300)

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
        fig.savefig(self.save_folder + "/indiv_rajectory.png", dpi=300)

    #--------------------------------------------------------
    # DISTRIBUTIONS
    #--------------------------------------------------------

    def plot_pdf_T_inst(self, iteration):    

        # Loading solution at given iteration
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')
        data = h5file_r.get(f"ITERATION_{iteration:05d}/all_states")[()]
        col_names = h5file_r[f"ITERATION_{iteration:05d}/all_states"].attrs["cols"]
        h5file_r.close()
        df = pd.DataFrame(data=data, columns=col_names)
            
        # Temperature histogram
        fig, ax = plt.subplots()
        
        sns.histplot(data=df, x="Temperature", ax=ax, stat="probability",
                     binwidth=20, kde=True)

        fig.tight_layout()
            
        fig.savefig(self.save_folder + f"/PDF_T_plot_iteration{iteration:05d}.png")
            
    def plot_pdf_T_all(self): 
            
        # Temperature histogram
        fig, ax = plt.subplots()
        
        sns.histplot(data=self.df, x="Temperature", ax=ax, stat="probability",
                     binwidth=20, kde=True)

        fig.tight_layout()
            
        fig.savefig(self.save_folder + "/PDF_T_plot.png")

    def plot_pdf_HRR_inst(self, iteration):    

        # Loading solution at given iteration
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')
        data = h5file_r.get(f"ITERATION_{iteration:05d}/all_states")[()]
        col_names = h5file_r[f"ITERATION_{iteration:05d}/all_states"].attrs["cols"]
        h5file_r.close()
        df = pd.DataFrame(data=data, columns=col_names)
            
        # Temperature histogram
        fig, ax = plt.subplots()
        
        sns.histplot(data=df, x="HRR", ax=ax, stat="probability",
                     binwidth=20, kde=True)

        fig.tight_layout()
            
        fig.savefig(self.save_folder + f"/PDF_HRR_plot_iteration{iteration:05d}.png")

    def plot_pdf_HRR_all(self): 
            
        # Temperature histogram
        fig, ax = plt.subplots()
        
        sns.histplot(data=self.df, x="log_abs_HRR", ax=ax, stat="probability",
                     bins=20000, kde=False)

        ax.set_xlim([-20.0,25])
        ax.set_ylim([0,0.002])

        fig.tight_layout()
            
        fig.savefig(self.save_folder + "/PDF_HRR_plot.png")

    #--------------------------------------------------------
    # Points density
    #--------------------------------------------------------

    def density_scatter(self, var_x , var_y, sort = True, bins = 100):
        # Functions from https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib

        x = self.df[var_x]
        y = self.df[var_y]

        fig , ax = plt.subplots()
        data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
        z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

        #To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0

        # Sort the points by density, so that the densest points are plotted last
        if sort :
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

        ax.scatter(x, y, c=z)

        norm = Normalize(vmin = np.min(z), vmax = np.max(z))
        cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
        cbar.ax.set_ylabel('Density')

        fig.tight_layout()

        fig.savefig(self.save_folder + f"/pnts_density_x{var_x}_y{var_y}_plot.png")