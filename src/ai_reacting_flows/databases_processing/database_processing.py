"""Database processing utilities for AI reacting flows.

This module exposes the :class:`LearningDatabase` class used to read raw
stochastic/flamelet databases, apply filtering, clustering, resampling and
produce machine‑learning ready train/validation datasets.
"""

import os
import sys
import shutil
import pickle

import oyaml as yaml
import joblib
import pandas as pd
import numpy as np
from scipy.interpolate import interpn
# from scipy.interpolate import interp1d
# from scipy.stats.kde import gaussian_kde
import h5py

# import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize 
from matplotlib import cm

import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler #, MinMaxScaler
from sklearn.model_selection import train_test_split

import ai_reacting_flows.tools.utilities as utils

sns.set_style("darkgrid")


class LearningDatabase(object):
    """Main entry point for database processing.

    The class reads raw HDF5 databases, optionally applies temperature
    thresholds, clustering and resampling, and finally produces
    train/validation datasets along with scalers and diagnostic plots.
    """

    def __init__(self):

        # Input parameters
        self.run_folder = os.getcwd()
        with open(os.path.join(self.run_folder, "dtb_processing.yaml"), "r") as file:
            dtb_processing_parameters = yaml.safe_load(file)

        database_params = dtb_processing_parameters["database_params"]
        self.database_type = database_params["database_type"]
        self.database_name = database_params["database_name"]
        self.input_dtb_file = database_params["dtb_file"]
        self.dt_var = database_params['dt_var']
        self.detailed_mechanism  = database_params["mech_file"]
        self.fuel  = database_params["fuel"]

        data_processing = dtb_processing_parameters["data_processing"]
        self.log_transform_X  = data_processing["log_transform_X"]
        self.log_transform_Y  = data_processing["log_transform_Y"]
        self.threshold  = data_processing["threshold"]
        self.T_threshold  = data_processing["T_threshold"]
        self.output_omegas  = data_processing["output_omegas"]
        self.with_N_chemistry = data_processing["with_N_chemistry"]

        data_clustering = dtb_processing_parameters["data_clustering"]
        self.clusterize_on = data_clustering['clusterize_on']
        self.clustering_method = data_clustering["clustering_method"]

        self.train_set_size = dtb_processing_parameters['train_set_size']

        self.nb_clusters = 1  # default value before clustering
        self.nb_clusters_input = data_clustering["nb_clusters"] # storing for later, if clustering is actually called

        if self.database_type not in ["stoch", "flamelets"]:
            sys.exit("Error on database_type. It should be 'stoch' or 'flamelets'")

        # Parameters depending on database type
        if self.database_type=="stoch":
            folder_prefix = "STOCH"
        elif self.database_type=="flamelets":
            folder_prefix = "FLAMELETS"
        
        # Folder where results are stored
        self.dtb_folder = f"{self.run_folder:s}/{folder_prefix}_DTB_" + database_params["dtb_folder_suffix"]


        # Check if mechanism is in YAML format
        if not self.detailed_mechanism.endswith("yaml"):
            sys.exit("ERROR: chemical mechanism should be in yaml format !")

        # Read H5 files to get databases in pandas format
        with h5py.File(os.path.join(self.dtb_folder, self.input_dtb_file), 'r') as h5file_r:
            names = h5file_r.keys()
            self.nb_solutions = len(names)
        self.get_database_from_h5()

        # Extracting some information
        self.species_names = self.X.columns[2:-2] # -2 because progvar and HRR
        print(f" >> species names: {self.species_names}")
        self.nb_species = len(self.species_names)

        # Saving folder: by default the model is saved in cluster0
        if os.path.isdir(self.dtb_folder + "/" + self.database_name):
            print(f">> Processed dataset {self.database_name} already exists => deleting")
            shutil.rmtree(self.dtb_folder + "/" + self.database_name)  # We remove previous generated case if still here
        os.mkdir(self.dtb_folder + "/" + self.database_name)

        # By default, we attribute cluster 0 to everyone
        self.X["cluster"] = 0.0

        # Clusterized datasets
        self.clusterized_dataset = False

        # Flag to check if database has already been processed
        self.is_processed = False

        # Default BCT parameter
        self.lambda_bct = 0.1

        # Saving detailed mechanism in database folder
        shutil.copy(self.detailed_mechanism, self.dtb_folder + "/" + self.database_name + "/mech_detailed.yaml")
        shutil.copy(os.path.join(self.run_folder, "dtb_processing.yaml"), self.dtb_folder + "/" + self.database_name + "/dtb_processing.yaml")

        # self.is_reduced = False
        self.is_resampled = False

        self.is_pca_computed = False

        self.check_inputs()

    
    def get_database_from_h5(self):

        print(f">> Reading database in file {os.path.join(self.dtb_folder, self.input_dtb_file)}")

        # Opening h5 file
        h5file_r = h5py.File(os.path.join(self.dtb_folder, self.input_dtb_file), 'r')

        # Solution 0 read to get columns names
        self.col_names_X = h5file_r["ITERATION_00000/X"].attrs["cols"]
        self.col_names_Y = h5file_r["ITERATION_00000/Y"].attrs["cols"]

        list_df_X = []
        list_df_Y = []
        if self.dt_var:
            list_dt_arrays = []

        # Loop on solutions
        for i in range(self.nb_solutions):

            if i%100==0:
                print(f"Opening solution: {i} / {self.nb_solutions}")

            data_X = h5file_r.get(f"ITERATION_{i:05d}/X")[()]
            data_Y = h5file_r.get(f"ITERATION_{i:05d}/Y")[()]
            if self.dt_var:
                data_dt = h5file_r.get(f"ITERATION_{i:05d}/DT")[()]

            list_df_X.append(pd.DataFrame(data=data_X, columns=self.col_names_X))
            if self.dt_var:
                list_dt_arrays.append(data_dt)
                list_df_Y.append(data_Y)    # 3D array cannot be stored in pandas dataframe
            else:
                list_df_Y.append(pd.DataFrame(data=data_Y, columns=self.col_names_Y))

        h5file_r.close()

        print("\n Performing concatenation of dataframes...")

        self.X = pd.concat(list_df_X, ignore_index=True)
        if self.dt_var:
            self.Y = np.concatenate(list_df_Y, axis=0)
            self.dt_array = np.concatenate(list_dt_arrays, axis=0)
        else:
            self.Y = pd.concat(list_df_Y, ignore_index=True)

        print("End of concatenation ! \n")


    def database_to_h5(self, file_path, file_name):
        """Export current X/Y to a single-iteration HDF5 file.

        Parameters
        ----------
        file_path : str
            Directory where the file should be created.
        file_name : str
            Name of the HDF5 file to create.
        """

        # Read column names from the original database
        with h5py.File(os.path.join(self.dtb_folder, self.input_dtb_file), 'r') as h5file_r:
            col_names_X = h5file_r["ITERATION_00000/X"].attrs["cols"]
            col_names_Y = h5file_r["ITERATION_00000/Y"].attrs["cols"]

        # Create output file
        with h5py.File(os.path.join(file_path, file_name), 'w') as h5file_w:
            grp = h5file_w.create_group(f"ITERATION_{0:05d}")

            # Drop internal columns (cluster label, PCA coordinates) if present.
            cols_to_drop = [c for c in ["cluster", "PC1", "PC2"] if c in self.X.columns]
            dset_X = grp.create_dataset('X', data=self.X.drop(columns=cols_to_drop))
            dset_X.attrs['cols'] = col_names_X

            dset_Y = grp.create_dataset('Y', data=self.Y)
            dset_Y.attrs['cols'] = col_names_Y

        print("\n H5PY dataset created")


    def apply_temperature_threshold(self):
        """
        Filter X and Y to keep only samples where Temperature exceeds T_threshold.
        Must be called before clustering.
        """

        if self.clusterized_dataset:
            raise RuntimeError("Temperature threshold should be performed before clustering")

        # Mask
        is_above_temp = self.X["Temperature"]>self.T_threshold

        if not is_above_temp.any():
            raise ValueError(f"No samples remain above T_threshold={self.T_threshold}")

         # Apply mask
        self.X = self.X[is_above_temp].reset_index(drop=True)

        if self.dt_var:
            self.Y = self.Y[is_above_temp.values, :]
            self.dt_array = self.dt_array[is_above_temp.values]
        else:
            self.Y = self.Y[is_above_temp].reset_index(drop=True)


    def clusterize_dataset(self, c_bounds=None):

        # Avoid mutable default for c_bounds
        if c_bounds is None:
            c_bounds = []

        self.clusterized_dataset = True

        # updated from 1 (before clustering) to its actually value read from inputs
        self.nb_clusters = self.nb_clusters_input

        # if self.clusterize_on == 'double':
        #     nb_clusters_phys, nb_clusters_time = self.nb_clusters
        #     self.nb_clusters_tot = nb_clusters_phys * nb_clusters_time
        # else :
        # self.nb_clusters_tot = self.nb_clusters

        assert (not self.is_processed)

        if self.clustering_method=="progvar":

            # assert self.clusterize_on != 'double'
            assert self.nb_clusters == len(c_bounds) - 1

            # Saving bounds for progress variables
            with open(self.dtb_folder + "/" + self.database_name + "/c_bounds.pkl", 'wb') as f:
                pickle.dump(c_bounds, f)

            # Assigning cluster can be made using pd.cut. CM: to check...
            self.X["cluster"] = pd.cut(
                                    self.X["Prog_var"],
                                    bins=c_bounds,
                                    labels=False,
                                    include_lowest=True,
                                    )

        elif self.clustering_method=="kmeans":
            
            spec_list = self.species_names.to_list()
            spec_list.insert(0,"Temperature")


            if self.with_N_chemistry is False:
                spec_list.remove("N2")

            if self.dt_var:
                if self.clusterize_on=="all":
                    spec_list.append("dt")
                    X_flat, _, _ = self._flatten_multi_dt_arrays(self.X, self.Y, self.dt_array)
                    data_kmeans = X_flat[spec_list].values.copy()
                elif self.clusterize_on=="dt":
                    _, _, data_kmeans = self._flatten_multi_dt_arrays(self.X, self.Y, self.dt_array)
                    data_kmeans = data_kmeans.reshape(-1, 1)
                elif self.clusterize_on=="phys":
                    data_kmeans = self.X[spec_list].values.copy()
                # elif self.clusterize_on=="double":
                #     data_kmeans_phys = self.X[spec_list].values.copy()
                #     data_kmeans_time = self.dt_array.copy()
            else:
                data_kmeans = self.X.values.copy()

            # Applying log transform
            if self.log_transform_X>0:
                if self.dt_var and self.clusterize_on=="dt":
                    data_kmeans = self._compute_log(data_kmeans)
                # elif self.dt_var and self.clusterize_on=="double":
                #     data_kmeans_phys[:, 1:] = self._compute_log(data_kmeans_phys[:, 1:])
                #     data_kmeans_time = self._compute_log(data_kmeans_time)
                else:
                    data_kmeans[:, 1:] = self._compute_log(data_kmeans[:, 1:])

            # Normalizing states before k-means
            # if self.dt_var and self.clusterize_on=="double":
            #     Xscaler = StandardScaler()
            #     Xscaler.fit(data_kmeans_phys)
            #     data_kmeans_phys = Xscaler.transform(data_kmeans_phys)
            #     #
            #     Tscaler = StandardScaler()
            #     Tscaler.fit(data_kmeans_time)
            #     data_kmeans_time = Tscaler.transform(data_kmeans_time)
            # else:
            Xscaler = StandardScaler()
            Xscaler.fit(data_kmeans)
            data_kmeans = Xscaler.transform(data_kmeans)

            # Applying k-means clustering
            print(">> Performing k-means clustering")
            # if self.dt_var and self.clusterize_on == "double":
            #     print("Shape (phys):", data_kmeans_phys.shape)
            #     print("Shape (time):", data_kmeans_time.shape)
            # else:
            print("Shape:", data_kmeans.shape)
            
            # if self.dt_var and self.clusterize_on=='double':
            #     kmeans_phys = KMeans(n_clusters=nb_clusters_phys, random_state=42).fit(data_kmeans_phys)
            #     kmeans_time = KMeans(n_clusters=nb_clusters_time, random_state=42).fit(data_kmeans_time)
            # else :
            kmeans = KMeans(n_clusters=self.nb_clusters, random_state=42).fit(data_kmeans)

            # if self.dt_var and self.clusterize_on=='double':
            #     print(kmeans_phys.labels_)
            #     print('KMeans Phys Score : ', kmeans_phys.score(data_kmeans_phys))
            #     print('KMeans Time Score : ', kmeans_time.score(data_kmeans_time))
            # else:
            print('KMeans Score : ', kmeans.score(data_kmeans))

            # Attributing cluster to data points
            # if self.dt_var and self.clusterize_on=='double':
            #     self.X["cluster_phys"] = kmeans_phys.labels_
            #     self.time_clusters = kmeans_time.labels_
            #     self.all_clusters = kmeans_phys.labels_ * nb_clusters_time + kmeans_time.labels_    #CM: BUG HERE array should be *n_dt  => to debug and double is OK
            if self.dt_var and self.clusterize_on=='dt':
                self.time_clusters = kmeans.labels_
            elif self.dt_var and self.clusterize_on=='all':
                self.all_clusters = kmeans.labels_
            else:  # constant dt or dt var with clusterize_on=phys
                self.X["cluster"] = kmeans.labels_

            # Saving K-means model
            # if self.dt_var and self.clusterize_on == 'double':
            #     with open(self.dtb_folder + "/" + self.database_name + "/kmeans_model_phys.pkl", "wb") as f:
            #         pickle.dump(kmeans_phys, f)
            #     with open(self.dtb_folder + "/" + self.database_name + "/kmeans_model_time.pkl", "wb") as f:
            #         pickle.dump(kmeans_time, f)
            # else:
            with open(self.dtb_folder + "/" + self.database_name + "/kmeans_model.pkl", "wb") as f:
                pickle.dump(kmeans, f)

            # Saving scaler
            joblib.dump(Xscaler, self.dtb_folder + "/" + self.database_name + "/Xscaler_kmeans.pkl")
            # if self.dt_var and self.clusterize_on == 'double':
            #     joblib.dump(Tscaler, self.dtb_folder + "/" + self.database_name + "/Tscaler_kmeans.pkl")

            # Saving normalization parameters and centroids
            np.savetxt(self.dtb_folder + "/" + self.database_name + '/kmeans_norm.dat', np.vstack([Xscaler.mean_, Xscaler.var_]).T)
            # if self.dt_var and self.clusterize_on == 'double':
            #     np.savetxt(self.dtb_folder + "/" + self.database_name + '/kmeans_norm_time.dat', np.vstack([Tscaler.mean_, Tscaler.var_]).T)
            #     centroids_phys = kmeans_phys.cluster_centers_
            #     centroids_time = kmeans_time.cluster_centers_
            #     centroids = np.zeros((self.nb_clusters_tot, len(centroids_phys[0])))
            #     k = 0
            #     for i in range(nb_clusters_phys):
            #         for j in range(nb_clusters_time):
            #             centroids[k, :] = np.append(centroids_phys[i, :-1], centroids_time[j, -1])
            #             k += 1
            #     np.savetxt(self.dtb_folder + "/" + self.database_name + '/km_centroids.dat', centroids.T)
            # else:
            np.savetxt(self.dtb_folder + "/" + self.database_name + '/km_centroids.dat', kmeans.cluster_centers_.T)



    def _compute_log(self, data):

        data_log = data.copy()

        data_log[data_log < self.threshold] = self.threshold
        if self.log_transform_X==1:
            data_log = np.log(data_log)
        elif self.log_transform_X==2:
            data_log = (data_log**self.lambda_bct - 1.0)/self.lambda_bct

        return data_log


    def visualize_clusters(self, var1, var2):

        # if self.dt_var and self.clusterize_on=="double":
        #     print("WARNING: double (physical, time) clustering is used; this function only shows the physical clustering.")
        #     cluster_var = "cluster_phys"
        # else:
        cluster_var = "cluster"

        if self.dt_var and (self.clusterize_on in ["dt", "all"]):
            raise ValueError("Current cluster visualization is only possible for physical clustering, 'all' and 'dt' are not allowed.")

        # If variables are PC1 and PC2, we compute the PCA
        if var1=="PC1" and var2=="PC2":
            if not self.is_pca_computed:
                self.compute_pca()

        fig, ax = plt.subplots()
        im = ax.scatter(self.X[var1], self.X[var2], c = self.X[cluster_var])
        fig.colorbar(im, ax=ax)
        ax.set_xlabel(var1, fontsize=16)
        ax.set_ylabel(var2, fontsize=16)
        fig.tight_layout()
        fig.savefig(self.dtb_folder + "/" + self.database_name + f"/cluster_{var1}_{var2}.png")


    # Re-sampling based on heat release rate
    def undersample_HRR(self, jpdf_var_1, jpdf_var_2, hrr_func, keep_low_c, n_samples=None, n_bins=100, seed=1991, plot_distrib=False):

        if self.clusterized_dataset and self.dt_var and self.clusterize_on in ("dt", "all"):
            raise NotImplementedError(
                "undersample_HRR does not currently support resampling after "
                "clustering on 'dt' or 'all' with dt_var=True, because the "
                "resulting cluster-label arrays would no longer match the "
                "resampled data. Cluster on 'phys' before resampling, or "
                "re-cluster after calling undersample_HRR."
            )
        
        # Save initial states for later use
        self.X_old = self.X.copy()
        self.Y_old = self.Y.copy()

        # If variables are PC1 and PC2, the re-sampling is done in (normalized) PC-space and we need to perform the PA
        if jpdf_var_1=="PC1" and jpdf_var_2=="PC2":
            if not self.is_pca_computed:
                self.compute_pca()

        # Dataset size
        n = self.X.shape[0]

        # Variable on which to take statistics
        a = np.abs(self.X["HRR"])
        a = (a-a.min())/(a.max()-a.min())
        a = hrr_func(a)

        # Setting a numpy seed
        np.random.seed(seed) 

        x = self.X[jpdf_var_1]
        y = self.X[jpdf_var_2]
        data, x_e, y_e = np.histogram2d(x, y, bins = n_bins, density = True)
        # data = data.T

        # fig, ax = plt.subplots()
        # ax.scatter(x, y, c=a, s=1)
        # norm = Normalize(vmin = np.min(a), vmax = np.max(a))
        # cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
        # cbar.ax.set_ylabel('logHRR')
        # fig.tight_layout()

        data_hrr, _, _ = np.histogram2d(x, y, bins = n_bins, density = True, weights=a)
        # data_hrr = data_hrr.T
        counts, _, _ = np.histogram2d(x, y, bins = n_bins, density = False)
        data_hrr = data_hrr / counts
        data_hrr[np.where(np.isnan(data_hrr))] = 0.0
        data_hrr[np.where(np.isinf(data_hrr))] = 0.0

        # METHOD OF CHI, A PRIORI NOT USEFUL
        # # Finding number of samples to keep (methodology of Chi et al.), if no number is provided in function input
        # if n_samples is None:
        #     # We first locate the bin with maximal HRR
        #     list_indices = np.where(data_hrr==np.amax(data_hrr))
        #     # We then compute the number of desired samples
        #     n_samples = n * (data[list_indices[0][0],list_indices[1][0]]/data_hrr[list_indices[0][0],list_indices[1][0]])
        #     n_samples = int(n_samples)

        # Computing weighting function
        f_m = (data_hrr/data) #(n_samples/n) * (data_hrr/data) (factor n_samples/n useless as we normalize later)
        f_m[np.where(np.isnan(f_m))] = 0.0

        # X, Y = np.meshgrid(x_e, y_e)
        # fig, ax = plt.subplots()
        # im = ax.pcolormesh(X, Y, data_hrr.T, cmap="viridis", vmin=0.0, vmax=0.0005)
        # fig.colorbar(im, ax=ax)
        # fig.show()

        # X, Y = np.meshgrid(x_e, y_e)
        # fig, ax = plt.subplots()
        # im=ax.pcolormesh(X, Y, data.T, cmap="viridis", vmin=0.0, vmax=0.1)
        # fig.colorbar(im, ax=ax)
        # fig.show()

        # p_1 = f_m/f_m.sum()
        # X, Y = np.meshgrid(x_e, y_e)
        # fig, ax = plt.subplots()
        # im = ax.pcolormesh(X, Y, p_1.T, cmap="viridis", vmin=0.0, vmax=0.000001)
        # fig.colorbar(im, ax=ax)
        # fig.show()

        f_m_interp = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , f_m , np.vstack([x,y]).T , method = "linear", bounds_error = False)
        f_m_interp[np.where(np.isnan(f_m_interp))] = 0.0

        # random.choice needs probability (sum to 1)
        p = f_m_interp/f_m_interp.sum()


        # PLOTTING FOR PAPER----------------------------------------------------------------------------

        data_hrr_interp = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data_hrr , np.vstack([x,y]).T , method = "linear", bounds_error = False)
        # data_hrr_interp[np.where(np.isnan(data_hrr_interp))] = 0.0
        #
        data_interp = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "linear", bounds_error = False)
        # data_interp[np.where(np.isnan(data_interp))] = 0.0

        # # Single plot (better for ppt)
        # fig, ax = plt.subplots()
        # im = ax.scatter(x, y, c=np.log(p), s=4, vmin=-20, vmax=-5)
        # cbar = fig.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel(r'$\log(f_m)$ $[-]$', fontsize=16)
        # fig.tight_layout()
        # ax.set_ylabel(r"$Y_{H_{2}O}$ $[-]$", fontsize=16)
        # ax.set_xlabel(r"$T$ $[K]$", fontsize=16)
        # fig.tight_layout()
        # fig.savefig("fm.png", dpi=400, bbox_inches="tight")

        if plot_distrib:
            fig, (ax1,ax2,ax3) = plt.subplots(nrows=3)
            fig.set_size_inches(18.5, 10.5)

            plt.set_cmap('plasma')
            im = ax1.scatter(x, y, c=data_interp, s=4, vmin=0, vmax=0.1)
            cbar = fig.colorbar(im, ax=ax1)
            cbar.ax.set_ylabel(r'$f_s$ $[-]$', fontsize=16)
            fig.tight_layout()

            im = ax2.scatter(x, y, c=np.log(data_hrr_interp), s=4, vmin=-10, vmax=np.log(0.00004))
            cbar = fig.colorbar(im, ax=ax2)
            cbar.ax.set_ylabel(r'$\log(f_q)$ $[-]$', fontsize=16)
            fig.tight_layout()

            im = ax3.scatter(x, y, c=np.log(p), s=4, vmin=-20, vmax=-5)
            cbar = fig.colorbar(im, ax=ax3)
            cbar.ax.set_ylabel(r'$\log(f_m)$ $[-]$', fontsize=16)
            fig.tight_layout()

            ax1.xaxis.set_ticklabels([])
            ax2.xaxis.set_ticklabels([])
            for ax in [ax1,ax2,ax3]:
                ax.set_box_aspect(1)
                # ax.set_ylabel(r"$Y_{H_{2}O}$ $[-]$", fontsize=16)
                ax.set_ylabel(jpdf_var_2, fontsize=16)
            # ax3.set_xlabel(r"$T$ $[K]$", fontsize=16)
            ax3.set_xlabel(jpdf_var_1, fontsize=16)

            fig.tight_layout()
            fig.savefig(os.path.join(self.dtb_folder, "resampling_analysis.png"), dpi=400, bbox_inches="tight")

        # -------------------------------------------------------------------------------------------------

        # Copying X to manipulate weights
        X_save = self.X.copy()
        X_save["p"] = p

        # Setting probability of points with p=0.0 to a very small value, so that we are able to reach the original datasets for high values of samples
        X_save.loc[X_save["p"]==0.0, 'p'] = X_save["p"].min()/1000.0

        # Imposing small progress variable values to stay in dataset (needed for ignition)
        if keep_low_c:
            X_save.loc[(X_save["Prog_var"]<0.2) & (X_save["Temperature"]>1000.0), 'p'] = 1.0

        # Normalizing
        p = X_save["p"]/X_save["p"].sum()

        # Performing the random choice of points
        choice = np.random.choice(range(n), replace=False, size=n_samples, p=p)

        # Performing the point selection in database
        self.X = self.X.iloc[choice]
        self.X = self.X.reset_index(drop=True)
        #
        if self.dt_var:
            self.Y = self.Y[choice, :]
            self.dt_array = self.dt_array[choice]
        else:
            self.Y = self.Y.iloc[choice]
            self.Y = self.Y.reset_index(drop=True)


        self.is_resampled = True

        # We recompute PCA because database has changed
        if jpdf_var_1=="PC1" and jpdf_var_2=="PC2":
            self.compute_pca()
    
        print(f"\n Number of points in undersampled dataset: {self.X.shape[0]} \n")
        print(f"    >> {100*self.X.shape[0]/n} % of the database is retained")


    # Database final processing
    def process_database(self, plot_distributions = False, distribution_species=[], seed = 42):

        self.is_processed = True

        # Also writing all data in a h5 file which is easier to read than csv. CM: maybe remove csv ultimately
        h5file_w = h5py.File(os.path.join(self.dtb_folder,self.database_name,"training_data.h5"), 'w')

        # Pressure removed by default (can be changed later)
        remove_pressure_X = True
        remove_pressure_Y = True

        # If variable dt is used, self.X is of size (N,n_var) and self.Y is shape (N,n_var,n_dt)
        # We flatten arrays to get both self.X and self.Y of shapes (n_dt*N,n_var)
        # We can then use same lines of code as the non variable dt, it should have no impact on the scalers.
        if self.dt_var:

            X, Y, _ = self._flatten_multi_dt_arrays(self.X, self.Y, self.dt_array)

            # Appending cluster indices in cases where it only makes sense on flatted arrays
            if self.clusterized_dataset: 
                if self.clusterize_on=="dt":
                    X["cluster"] = self.time_clusters   #in this case clustering is made on dt only
                elif self.clusterize_on=="all": # or self.clusterize_on=="double":
                    X["cluster"] = self.all_clusters

        # Constant dt 
        else:
            X = self.X.copy()
            Y = self.Y.copy()

        for i_cluster in range(self.nb_clusters):
            
            print("", flush=True)
            print(f"CLUSTER {i_cluster}:", flush=True)

            # Isolate cluster i
            X_p = X[X["cluster"]==i_cluster]
            if self.dt_var:
                Y_p = Y.loc[X["cluster"]==i_cluster, :]
            else:
                Y_p = Y[X["cluster"]==i_cluster]

            # Reset indexes
            X_p = X_p.reset_index(drop=True)
            Y_p = Y_p.reset_index(drop=True)

            # Removing useless columns
            X_cols = X_p.columns.to_list()
            # if self.clusterized_dataset and self.dt_var and self.clusterize_on == 'double': 
            #     list_to_remove = ["Prog_var", "HRR", "cluster", "cluster_phys"]
            # else:
            list_to_remove = ["Prog_var", "HRR", "cluster"]
            
            if remove_pressure_X:
                list_to_remove.append('Pressure')
            if self.is_pca_computed:
                list_to_remove.append("PC1")
                list_to_remove.append("PC2")
            #
            [X_cols.remove(elt) for elt in list_to_remove]   
            X_p = X_p[X_cols] 
            #
            Y_cols = Y_p.columns.to_list()
            list_to_remove = ["Temperature", "Prog_var", "HRR"]
            
            if remove_pressure_Y:
                list_to_remove.append('Pressure')
            #
            [Y_cols.remove(elt) for elt in list_to_remove]   
            Y_p = Y_p[Y_cols] 
            

            # Clip if logarithm transformation
            if self.log_transform_X>0:
                X_p[X_p < self.threshold] = self.threshold
            if self.log_transform_Y>0:
                Y_p[Y_p < self.threshold] = self.threshold

            # If log transform at input and not at output and output_omega, we need to save un-transformed data
            if self.log_transform_X>0 and self.log_transform_Y==0 and self.output_omegas:
                if self.dt_var:
                    X_p_save = X_p.loc[:, X_cols[1:-1]] # dt should not be taken into account
                else:
                    X_p_save = X_p.loc[:, X_cols[1:]]

            # Applying transformation (log of BCT)
            if self.log_transform_X==1:
                X_p.loc[:, X_cols[1:]] = np.log(X_p[X_cols[1:]])
            elif self.log_transform_X==2:
                X_p.loc[:, X_cols[1:]] = (X_p[X_cols[1:]]**self.lambda_bct - 1.0)/self.lambda_bct
            #
            if self.log_transform_Y==1:
                Y_p.loc[:, Y_cols] = np.log(Y_p[Y_cols])
            elif self.log_transform_Y==2:
                Y_p.loc[:, Y_cols] = (Y_p[Y_cols]**self.lambda_bct - 1.0)/self.lambda_bct


            # If differences are considered
            if self.output_omegas:
                if self.log_transform_X>0 and self.log_transform_Y==0:  # Case where X has been logged but not Y
                    Y_p = Y_p.subtract(X_p_save.reset_index(drop=True))
                else:
                    if self.dt_var:
                        Y_p = Y_p.subtract(X_p.loc[:,X_cols[1:-1]].reset_index(drop=True))  # dt should not be taken into account
                    else:
                        Y_p = Y_p.subtract(X_p.loc[:,X_cols[1:]].reset_index(drop=True))

            # Renaming columns
            X_p.columns = [str(col) + '_X' for col in X_p.columns]
            Y_p.columns = [str(col) + '_Y' for col in Y_p.columns]

            if not self.with_N_chemistry:
                X_p = X_p.drop("N2_X", axis=1)
                Y_p = Y_p.drop("N2_Y", axis=1)


            # Train validation split
            X_train, X_val, Y_train, Y_val = train_test_split(X_p, Y_p, train_size=self.train_set_size, random_state=seed)

            # === SCALERS ===
            # NORMALIZING X
            Xscaler = StandardScaler()
            X_train_array = Xscaler.fit_transform(X_train)
            X_val_array = Xscaler.transform(X_val)

            X_train = pd.DataFrame(X_train_array, columns=X_train.columns, index=X_train.index)
            X_val = pd.DataFrame(X_val_array, columns=X_val.columns, index=X_val.index)
            
            # NORMALIZING Y
            Yscaler = StandardScaler()
            Y_train_array = Yscaler.fit_transform(Y_train)
            Y_val_array = Yscaler.transform(Y_val)

            Y_train = pd.DataFrame(Y_train_array, columns=Y_train.columns, index=Y_train.index)
            Y_val = pd.DataFrame(Y_val_array, columns=Y_val.columns, index=Y_val.index)
        

            # Forcing constant N2
            # n2_cte = not self.with_N_chemistry
            # if n2_cte:
            #     if self.output_omegas:
            #         Y_train["N2_Y"] = 0.0
            #         Y_val["N2_Y"] = 0.0
            #     else:
            #         Y_train["N2_Y"] = X_train["N2_X"]
            #         Y_val["N2_Y"] = X_val["N2_X"]

            print(">> Saving datasets", flush=True)

            # Saving in h5 format
            grp = h5file_w.create_group(f"CLUSTER_{i_cluster}")

            # Saving scalers
            Xscaler_array = np.column_stack([Xscaler.mean_, Xscaler.var_])
            dset_Xscaler = grp.create_dataset('Xscaler', data = Xscaler_array)
            dset_Xscaler.attrs['cols'] = np.array(["mean", "std"], dtype=object)
            #
            Yscaler_array = np.column_stack([Yscaler.mean_, Yscaler.var_])
            dset_Yscaler = grp.create_dataset('Yscaler', data = Yscaler_array)
            dset_Yscaler.attrs['cols'] = np.array(["mean", "std"], dtype=object)
            
            # Saving data
            dset_X_train = grp.create_dataset('X_train', data = X_train)
            dset_X_train.attrs['cols'] = np.array(X_train.columns, dtype=object)
            #
            dset_Y_train = grp.create_dataset('Y_train', data = Y_train)
            dset_Y_train.attrs['cols'] = np.array(Y_train.columns, dtype=object)
            #
            dset_X_val = grp.create_dataset('X_val', data = X_val)
            dset_X_val.attrs['cols'] = np.array(X_val.columns, dtype=object)
            #
            dset_Y_val = grp.create_dataset('Y_val', data = Y_val)
            dset_Y_val.attrs['cols'] = np.array(Y_val.columns, dtype=object)


            # Saving datasets
            # print(">> Saving datasets")
            # X_train.to_csv(self.dtb_folder + "/" + self.database_name + f"/cluster{i_cluster}/X_train.csv", index=False)
            # X_val.to_csv(self.dtb_folder + "/" + self.database_name + f"/cluster{i_cluster}/X_val.csv", index=False)
            # Y_train.to_csv(self.dtb_folder + "/" + self.database_name + f"/cluster{i_cluster}/Y_train.csv", index=False)
            # Y_val.to_csv(self.dtb_folder + "/" + self.database_name + f"/cluster{i_cluster}/Y_val.csv", index=False)

            # Plotting distributions if necessary
            if plot_distributions:
                print(">> Plotting distributions", flush=True)
                for spec in distribution_species:
                   self.plot_distributions(X_train, X_val, Y_train, Y_val, spec, i_cluster)

        h5file_w.close()


    def _flatten_multi_dt_arrays(self, X_p, Y_p, dt_p):     
        
        # X columns names (it is still a pandas dataframe)
        #Keeping columns names at the end of this routine enables a transparent removal of variables later in process_database
        col_names_X = X_p.columns.tolist()
        
        N = X_p.shape[0]
        n_var_X = X_p.shape[1]
        n_var_Y = Y_p.shape[1]
        n_dt = dt_p.shape[1]

        # Flatten dt array 
        dt_p_flat = dt_p.flatten()

        X_p_flat_arr = np.empty((n_dt*N,n_var_X+1))
        for i_dt in range(n_dt):
            X_p_flat_arr[i_dt::n_dt,:-1] = X_p.values    # X_p is pandas
        X_p_flat_arr[:,-1] = dt_p_flat

        Y_p_flat_arr = np.empty((n_dt*N,n_var_Y))
        for i_dt in range(n_dt):
            Y_p_flat_arr[i_dt::n_dt,:] = Y_p[:,:,i_dt]

        # We finally want pandas dataframes
        col_names_X.append("dt")
        X_p_flat = pd.DataFrame(data=X_p_flat_arr, columns=col_names_X)
        #
        Y_p_flat = pd.DataFrame(data=Y_p_flat_arr, columns=self.col_names_Y)   # We assume no additional columns have been added to Y after reading h5

        return X_p_flat, Y_p_flat, dt_p_flat


    def print_data_size(self):

        # Printing number of elements for each class
        size_total = 0

        if self.dt_var and self.clusterize_on == "dt":
            for i in range(self.nb_clusters):

                size_cluster_i = self.time_clusters[self.time_clusters==i].shape[0]
                size_total += size_cluster_i

                print(f">> There are {size_cluster_i} points in cluster{i}")

        # elif self.dt_var and self.clusterize_on in ["all", "double"]:
        #     for i in range(self.nb_clusters_tot):
        #         size_cluster_i = self.all_clusters[self.all_clusters==i].shape[0]
        #         size_total += size_cluster_i

        #         print(f">> There are {size_cluster_i} points in cluster{i}")

        else:
            for i in range(self.nb_clusters):

                size_cluster_i = self.X[self.X["cluster"]==i].shape[0]
                size_total += size_cluster_i

                print(f">> There are {size_cluster_i} points in cluster{i}")

            print(f"\n => There are {size_total} points overall")

    
    # Distribution plotting function
    def plot_distributions(self, X_train, X_val, Y_train, Y_val, species, i_cluster):

        # Number of collocations points for pdf
        n = 400
        
        # Computing PDF's
        x_train = np.linspace(X_train[species + "_X"].min(), X_train[species + "_X"].max(), n)
        pdf_X_train = utils.compute_pdf(x_train, X_train[species + "_X"])

        x_val = np.linspace(X_val[species + "_X"].min(), X_val[species + "_X"].max(), n)
        pdf_X_val = utils.compute_pdf(x_val, X_val[species + "_X"])

        if species!="Temperature":
            y_train = np.linspace(Y_train[species + "_Y"].min(), Y_train[species + "_Y"].max(), n)
            pdf_Y_train = utils.compute_pdf(y_train, Y_train[species + "_Y"])

            y_val = np.linspace(Y_val[species + "_Y"].min(), Y_val[species + "_Y"].max(), n)
            pdf_Y_val = utils.compute_pdf(y_val, Y_val[species + "_Y"])

        fig1, (ax1, ax2) = plt.subplots(ncols=2)
        fig2, (ax3, ax4) = plt.subplots(ncols=2)

        ax1.plot(x_train, pdf_X_train, color="k", lw=2)
        ax1.set_xlabel(f"{species} X $[-]$", fontsize=12)
        ax1.set_ylabel("pdf $[-]$", fontsize=12)
        #
        if species!="Temperature":
            ax2.plot(y_train, pdf_Y_train, color="k", lw=2)
            ax2.set_xlabel(f"{species} Y $[-]$", fontsize=12)
            ax2.set_ylabel("pdf $[-]$", fontsize=12)

        ax3.plot(x_val, pdf_X_val, color="k", lw=2)
        ax3.set_xlabel(f"{species} X $[-]$", fontsize=12)
        ax3.set_ylabel("pdf $[-]$", fontsize=12)
        #
        if species!="Temperature":
            ax4.plot(y_val, pdf_Y_val, color="k", lw=2)
            ax4.set_xlabel(f"{species} Y $[-]$", fontsize=12)
            ax4.set_ylabel("pdf $[-]$", fontsize=12)

        for ax in [ax1,ax2,ax3,ax4]:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0),useMathText=True)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)

            ax.xaxis.get_offset_text().set_fontsize(10)
            ax.yaxis.get_offset_text().set_fontsize(10)

            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=8)

            ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

        fig1.tight_layout()
        fig2.tight_layout()

        fig1.savefig(self.dtb_folder + "/" + self.database_name + f"/distrib_{species}_train_cluster_{i_cluster}.png", dpi=300, bbox_inches='tight')
        fig2.savefig(self.dtb_folder + "/" + self.database_name + f"/distrib_{species}_val_cluster_{i_cluster}.png", dpi=300, bbox_inches='tight')

    def check_inputs(self):

        if self.log_transform_X==0 and self.log_transform_Y>0:
            sys.exit("ERROR: log_transform_Y cannot be active if log_transform_X==0 !")

    # 2-D joint PDF plotting of two variable
    def density_scatter(self, var_x , var_y, sort = True, bins = 100):
        # Functions from https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib

        # If variables are PC1 and PC2, we compute the PCA
        if var_x=="PC1" and var_y=="PC2":
            if not self.is_pca_computed:
                self.compute_pca()
            
        x = self.X[var_x]
        y = self.X[var_y]

        fig , ax = plt.subplots()
        data , x_e, y_e = np.histogram2d(x, y, bins = bins, density = False )
        z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

        # To be sure to plot all data
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

        if self.is_resampled:
            fig.savefig(os.path.join(self.dtb_folder, f"density_{var_x}_{var_y}_resampled.png"), dpi=400)
        else:
            fig.savefig(os.path.join(self.dtb_folder, f"density_{var_x}_{var_y}.png"), dpi=400)


    # Marginal PDF of a given variable in the dataframe
    def plot_pdf_var(self, var):

        if var=="PC1" or var=="PC2":
            if not self.is_pca_computed:
                self.compute_pca()
            
        # Temperature histogram
        fig, ax = plt.subplots()
        
        sns.histplot(data=self.X, x=var, ax=ax, stat="density",
                     binwidth=10, kde=False)

        ax.set_box_aspect(1)

        if var=="Temperature":
            ax.set_xlabel("$T$ $[K]$", fontsize=14)
        ax.set_ylabel("Point density $[-]$", fontsize=14)

        ax.legend(fontsize=10)

        fig.tight_layout()
        if self.is_resampled:
            fig.savefig(self.dtb_folder + f'_distribution_{var}_resampled.png', dpi=400)
        else:
            fig.savefig(self.dtb_folder + f'_distribution_{var}.png', dpi=400)



    # Compare
    def compare_resampled_pdfs(self, var): 

        if var=="PC1" or var=="PC2":
            if not self.is_pca_computed:
                self.compute_pca()

        if self.is_resampled is False:
            sys.exit("Data has not been resampled, this function is not accessible")
            
        # Temperature histogram
        fig, ax1 = plt.subplots()
        
        sns.histplot(data=self.X_old, x=var, ax=ax1, color="black", stat="density",
                     binwidth=10, kde=False, label="Before re-sampling")

        sns.histplot(data=self.X, x=var, ax=ax1, color="blue", stat="density",
                     binwidth=10, kde=False, alpha=0.4, label="After re-sampling")
        
        ax1.set_box_aspect(1)
        
        if var=="Temperature":
            ax1.set_xlabel("$T$ $[K]$", fontsize=14)
        ax1.set_ylabel("Point density $[-]$", fontsize=14)

        ax1.legend(fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(self.dtb_folder,"comparison_distribution.png"), dpi=400)

    # Compute PCA of database
    def compute_pca(self):

        # Number of PCA dimensions here forced to 2
        k = 2

        features = self.species_names.to_list()
        features.insert(0,"Temperature")
        if self.with_N_chemistry is False:
            features.remove("N2")

        # Get states only (temperature and Yk's)
        data = self.X[features].values

        # # If log transform, we also apply it for pca
        # if self.log_transform_X>0:
        #     data[data < self.threshold] = self.threshold
        #     if self.log_transform_X==1:
        #         data[:, 1:] = np.log(data[:, 1:])
        #     elif self.log_transform_X==2:
        #         data[:, 1:] = (data[:, 1:]**self.lambda_bct - 1.0)/self.lambda_bct

        # Scaling data
        pca_scaler = StandardScaler()
        pca_scaler.fit(data)
        data = pca_scaler.transform(data)

        # Performing PCA
        pca_algo = PCA(n_components=k, svd_solver="full")
        pca_algo.fit(data)
        PCs = pca_algo.transform(data)


        self.X["PC1"] = PCs[:,0]
        self.X["PC2"] = PCs[:,1]

        self.is_pca_computed = True
