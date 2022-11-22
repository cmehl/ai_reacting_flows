import os, sys
import shutil
import pickle
import shelve
import random

import joblib
import pandas as pd
import numpy as np
from scipy.interpolate import interpn
from scipy.interpolate import interp1d
from scipy.stats.kde import gaussian_kde
import h5py
import cantera as ct

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize 
from matplotlib import cm

import seaborn as sns
sns.set_style("darkgrid")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import ai_reacting_flows.tools.utilities as utils


class LearningDatabase(object):

    def __init__(self, dtb_processing_parameters):

        # Input parameters
        self.dtb_folder  = dtb_processing_parameters["dtb_folder"]
        self.database_name = dtb_processing_parameters["database_name"]
        self.log_transform_X  = dtb_processing_parameters["log_transform_X"]
        self.log_transform_Y  = dtb_processing_parameters["log_transform_Y"]
        self.threshold  = dtb_processing_parameters["threshold"]
        self.output_omegas  = dtb_processing_parameters["output_omegas"]
        self.detailed_mechanism  = dtb_processing_parameters["detailed_mechanism"]
        self.fuel  = dtb_processing_parameters["fuel"]
        self.with_N_chemistry = dtb_processing_parameters["with_N_chemistry"]

        # Check if mechanism is in YAML format
        if self.detailed_mechanism.endswith("yaml") is False:
            sys.exit("ERROR: chemical mechanism should be in yaml format !")


        # Read H5 files to get databases in pandas format
        h5file_r = h5py.File(self.dtb_folder + f"/solutions.h5", 'r')
        names = h5file_r.keys()
        self.nb_solutions = len(names)
        h5file_r.close()
        self.get_database_from_h5()

        # Extracting some information
        self.species_names = self.X.columns[2:-2]
        self.nb_species = len(self.species_names)

        # Saving folder: by default the model is saved in cluster0
        if os.path.isdir(self.dtb_folder + "/" + self.database_name):
            print(f">> Processed dataset {self.database_name} already exists => deleting")
            shutil.rmtree(self.dtb_folder + "/" + self.database_name)  # We remove previous generated case if still here
        os.mkdir(self.dtb_folder + "/" + self.database_name)
        os.mkdir(self.dtb_folder + "/" + self.database_name + "/cluster0")

        # Saving inputs in file to be read when building ANN
        shelfFile = shelve.open(self.dtb_folder + "/" + self.database_name + "/dtb_params")
        #
        shelfFile["threshold"] = self.threshold
        shelfFile["log_transform_X"] = self.log_transform_X
        shelfFile["log_transform_Y"] = self.log_transform_Y
        shelfFile["output_omegas"] = self.output_omegas
        shelfFile["with_N_chemistry"] = self.with_N_chemistry
        shelfFile["clusterization_method"] = None   # Default value, erased if clustering is done
        #
        shelfFile.close()

        # By default, we attribute cluster 0 to everyone
        self.X["cluster"] = 0.0

        # Clusterized datasets
        self.clusterized_dataset = False
        self.nb_clusters = 1

        self.is_processed = False

        # Default BCT parameter
        self.lambda_bct = 0.1

        # Saving detailed mechanism in database folder
        shutil.copy(self.detailed_mechanism, self.dtb_folder + "/" + self.database_name + "/mech_detailed.yaml")

        self.is_reduced = False

        self.check_inputs()

    
    def get_database_from_h5(self):

        # Opening h5 file
        h5file_r = h5py.File(self.dtb_folder + "/solutions.h5", 'r')

        # Solution 0 read to get columns names
        col_names_X = h5file_r["ITERATION_00000/X"].attrs["cols"]
        col_names_Y = h5file_r["ITERATION_00000/Y"].attrs["cols"]

        # Loop on solutions
        list_df_X = []
        list_df_Y = []
        for i in range(self.nb_solutions):

            if i%100==0:
                print(f"Opening solution: {i} / {self.nb_solutions}")

            data_X = h5file_r.get(f"ITERATION_{i:05d}/X")[()]
            data_Y = h5file_r.get(f"ITERATION_{i:05d}/Y")[()]

            list_df_X.append(pd.DataFrame(data=data_X, columns=col_names_X))
            list_df_Y.append(pd.DataFrame(data=data_Y, columns=col_names_Y))

        h5file_r.close()

        print(f"\n Performing concatenation of dataframes...")

        self.X = pd.concat(list_df_X, ignore_index=True)
        self.Y = pd.concat(list_df_Y, ignore_index=True)

        print("End of concatenation ! \n")


    def apply_temperature_threshold(self, T_threshold):

        if self.clusterized_dataset:
            sys.exit("ERROR: Temperature threshold should be performed before clustering")

        # Mask
        is_above_temp = self.X["Temperature"]>T_threshold

        # Apply mask
        self.X = self.X[is_above_temp]
        self.Y = self.Y[is_above_temp]

        # Reset indexes
        self.X = self.X.reset_index(drop=True)
        self.Y = self.Y.reset_index(drop=True)



    def clusterize_dataset(self, clusterization_method, nb_clusters, c_bounds = []):


        # Saving clustering method
        shelfFile = shelve.open(self.dtb_folder + "/" + self.database_name + "/dtb_params")
        #
        shelfFile["clusterization_method"] = clusterization_method
        #
        shelfFile.close()

        assert self.is_processed == False

        self.clusterized_dataset = True

        # Creating empty folders for storing datasets
        for i in range(1, nb_clusters):
            os.mkdir(self.dtb_folder + "/" + self.database_name + f"/cluster{i}")

        self.nb_clusters = nb_clusters

        if clusterization_method=="progvar":

            assert nb_clusters == len(c_bounds) - 1

            # Saving bounds for progress variables
            with open(self.dtb_folder + "/" + self.database_name + "/c_bounds.pkl", 'wb') as f:
                pickle.dump(c_bounds, f)

            # Attributing cluster
            for i in range(nb_clusters):

                #TODO Painful loop, must be written using pandas functions
                for index, row in self.X.iterrows():
                    c = row["Prog_var"]
                    if (c>=c_bounds[i]) & (c<c_bounds[i+1]):
                        row["cluster"] = i
                

        elif clusterization_method=="kmeans":

            # Get states only (temperature and Yk's)
            cols = [0] + [2+i for i in range(self.nb_species)]
            data = self.X.values[:,cols]

            # If log transform, we also apply it when clustering
            if self.log_transform_X>0:
                data[data < self.threshold] = self.threshold
                if self.log_transform_X==1:
                    data[:, 1:] = np.log(data[:, 1:])
                elif self.log_transform_X==2:
                    data[:, 1:] = (data[:, 1:]**self.lambda_bct - 1.0)/self.lambda_bct

            # Normalizing states before k-means
            Xscaler = StandardScaler()
            Xscaler.fit(data)
            data = Xscaler.transform(data)

            # Applying k-means clustering
            print(">> Performing k-means clustering")
            kmeans = KMeans(n_clusters=nb_clusters, random_state=42).fit(data)

            # Attributing cluster to data points
            self.X["cluster"] = kmeans.labels_

            # Saving K-means model
            with open(self.dtb_folder + "/" + self.database_name + "/kmeans_model.pkl", "wb") as f:
                pickle.dump(kmeans, f)

            # Saving scaler
            joblib.dump(Xscaler, self.dtb_folder + "/" + self.database_name + "/Xscaler_kmeans.pkl")



    def visualize_clusters(self, species):

        fig, ax = plt.subplots()
        im = ax.scatter(self.X["Temperature"], self.X[species], c = self.X["cluster"])
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("T [K]")
        ax.set_ylabel("Y H2 [-]")
        fig.tight_layout()
        fig.savefig(f"cluster_T_{species}.png")



    # Very basic re-sampling, not used
    def undersample_cluster(self, i_cluster, ratio_to_keep):
        
        # Here it is better to work with the ratio of data to delete
        ratio_to_delete = 1.0 - ratio_to_keep
        
        # Indices of the given cluster
        indices_cluster = self.X[self.X["cluster"]==i_cluster].index

        # Sampling random rows to delete
        random.seed(a=42)
        rows = random.sample(indices_cluster.tolist(),int(ratio_to_delete*len(indices_cluster)))

        # Delete the rows in X and Y dataframes
        self.X.drop(rows, axis=0, inplace=True)
        self.Y.drop(rows, axis=0, inplace=True)



    # Re-sampling based on heat release rate
    def undersample_HRR(self, jpdf_var_1, jpdf_var_2, n_samples=None, n_bins=100):
        
        # Dataset size
        n = self.X.shape[0]

        # Variable on which to take statistics
        a = np.abs(self.X["HRR"])
        a = (a-a.min())/(a.max()-a.min())
        a = 1.0 + 1.0*a

        # Setting a numpy seed
        np.random.seed(1991) 

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

        # Finding number of samples to keep (methodology of Chi et al.), if no number is provided in function input
        if n_samples is None:
            # We first locate the bin with maximal HRR
            list_indices = np.where(data_hrr==np.amax(data_hrr))
            # We then compute the number of desired samples
            n_samples = n * (data[list_indices[0][0],list_indices[1][0]]/data_hrr[list_indices[0][0],list_indices[1][0]])
            n_samples = int(n_samples)

        # Computing weighting function
        f_m = (n_samples/n) * (data_hrr/data)
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

        # fig, ax = plt.subplots()
        # im = ax.scatter(x, y, c=p, s=4, vmin=0.0, vmax=0.00005)
        # cbar = fig.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel('p')
        # fig.tight_layout()

        # We impose big weights for points where c<0.2
        X_save = self.X.copy()
        X_save["p"] = p
        X_save.loc[X_save["Prog_var"]<0.2, 'p'] = 1.0
        p = X_save["p"]/X_save["p"].sum()

        # Performing the random choice of points
        choice = np.random.choice(range(n), replace=False, size=n_samples, p=p)

        # Performing the point selection in database
        self.X = self.X.iloc[choice]
        self.X = self.X.reset_index(drop=True)
        #
        self.Y = self.Y.iloc[choice]
        self.Y = self.Y.reset_index(drop=True)

        print(f"\n Number of points in undersampled dataset: {self.X.shape[0]} \n")
        print(f"    >> {100*self.X.shape[0]/n} % of the database is retained")
    


    # Function to reduce the species in the database. A closure based on fictive species is here selected
    def reduce_species_set(self, species_subset, fictive_species):

        # Checking if database has not already been reduced
        if self.is_reduced:
            sys.exit("ERROR: database is already reduced !")

        # Number of species in each species set
        nb_spec_red = len(species_subset)
        nb_spec_fictive = len(fictive_species)

        # Cantera for full mechanism
        gas = ct.Solution(self.detailed_mechanism)
        species_detailed = gas.species_names

        # Getting matrix (Wj/Wk)*n_k^j   (Remark: order of atoms is C, H, O, N)
        A_atomic_detailed = utils.get_molar_mass_atomic_matrix(species_detailed, self.fuel, self.with_N_chemistry)
        
        # Getting atomic mass fractions before reduction
        Yk_in = self.X[gas.species_names].values
        Yk_out = self.Y[gas.species_names].values
        #
        Ya_in = np.dot(A_atomic_detailed, Yk_in.transpose()).transpose()
        Ya_out = np.dot(A_atomic_detailed, Yk_out.transpose()).transpose()

        # Getting enthalpies
        h_in = np.sum(gas.partial_molar_enthalpies/gas.molecular_weights*Yk_in, axis=1)
        h_out = np.sum(gas.partial_molar_enthalpies/gas.molecular_weights*Yk_out, axis=1)


        # Removing unwanted species
        to_keep = ["Temperature", "Pressure"] + species_subset + ["Prog_var", "HRR", "cluster"]
        self.X = self.X[to_keep]
        to_keep = ["Temperature", "Pressure"] + species_subset + ["Prog_var", "HRR"]
        self.Y = self.Y[to_keep]


        # Flag is changed here because databases have been changed
        self.is_reduced = True


        # Getting atomic mass fractions of the reduced set of mass fractions
        A_atomic_reduced = utils.get_molar_mass_atomic_matrix(species_subset, self.fuel, self.with_N_chemistry)
        #
        Yk_red_in = self.X[species_subset].values
        Yk_red_out = self.Y[species_subset].values
        #
        Ya_red_in = np.dot(A_atomic_reduced, Yk_red_in.transpose()).transpose()
        Ya_red_out = np.dot(A_atomic_reduced, Yk_red_out.transpose()).transpose()
        # 
        # We need reduced enthalpy and molecular weights vectors
        partial_molar_enthalpies_red = np.empty(nb_spec_red)
        molecular_weights_red = np.empty(nb_spec_red)
        for i, spec in enumerate(species_subset):
            partial_molar_enthalpies_red[i] = gas.partial_molar_enthalpies[gas.species_index(spec)]
            molecular_weights_red[i] = gas.molecular_weights[gas.species_index(spec)]
        #
        h_red_in = np.sum(partial_molar_enthalpies_red/molecular_weights_red*Yk_red_in, axis=1)
        h_red_out = np.sum(partial_molar_enthalpies_red/molecular_weights_red*Yk_red_out, axis=1)

        # Missing atomic mass and enthalpy
        Delta_Ya_in = Ya_in - Ya_red_in
        Delta_Ya_out = Ya_out - Ya_red_out
        #
        Delta_h_in = h_in - h_red_in
        Delta_h_out = h_out - h_red_out

        # Creating delta vector
        Delta_in = np.concatenate([Delta_Ya_in, Delta_h_in.reshape(-1,1)], axis=1)
        Delta_out = np.concatenate([Delta_Ya_out, Delta_h_out.reshape(-1,1)], axis=1)

        # Creating matrix
        A_atomic_fictive = utils.get_molar_mass_atomic_matrix(fictive_species, self.fuel, self.with_N_chemistry)
        # We need reduced enthalpy and molecular weights vectors for fictive species
        partial_molar_enthalpies_fictive = np.empty(nb_spec_fictive)
        molecular_weights_fictive = np.empty(nb_spec_fictive)
        for i, spec in enumerate(fictive_species):
            partial_molar_enthalpies_fictive[i] = gas.partial_molar_enthalpies[gas.species_index(spec)]
            molecular_weights_fictive[i] = gas.molecular_weights[gas.species_index(spec)]
        #
        delta_h_f = partial_molar_enthalpies_fictive/molecular_weights_fictive
        #
        matrix_linear_system = np.concatenate([A_atomic_fictive, delta_h_f.reshape(1,-1)])

        # Inverting matrix and solving system
        matrix_inv = np.linalg.inv(matrix_linear_system)

        # Getting mass fractions of fictive species
        Yk_in_fictive = np.dot(matrix_inv, Delta_in.transpose()).transpose()
        Yk_out_fictive = np.dot(matrix_inv, Delta_out.transpose()).transpose()

        # Filling database with new fictive species
        for i, spec in enumerate(fictive_species):
            self.X[spec + "_F"] = Yk_in_fictive[:,i]
            self.Y[spec + "_F"] = Yk_out_fictive[:,i]

        # Moving progress variable, HRR, and cluster at the end
        self.X = self.X[[c for c in self.X if c not in ['Prog_var', 'HRR', 'cluster']] + ['Prog_var', 'HRR', 'cluster']]
        self.Y = self.Y[[c for c in self.Y if c not in ['Prog_var', 'HRR']] + ['Prog_var', 'HRR']]

        # Modification of the CANTERA mechanism
        mechanism = utils.cantera_yaml(self.detailed_mechanism)
        mechanism.remove_reactions()
        mechanism.reduce_mechanism_and_add_fictive_species(species_subset, fictive_species, "_F")
        mechanism.export_to_yaml(self.dtb_folder + "/" + self.database_name + "/mech_reduced.yaml")


        # Updating species names for later consistency
        self.species_names = self.X.columns[2:-2]
        self.nb_species = len(self.species_names)

        #-----------------VERIFICATION---------------------------------
        # Gas with retained species and fictive species
        gas_red_mech = ct.Solution(self.dtb_folder + "/" + self.database_name + "/mech_reduced.yaml")

        # Getting atomic mass fractions
        A_atomic_new = utils.get_molar_mass_atomic_matrix(gas_red_mech.species_names, self.fuel, self.with_N_chemistry)
        #
        Yk_new_in = self.X[gas_red_mech.species_names].values
        Yk_new_out = self.Y[gas_red_mech.species_names].values
        #
        Ya_new_in = np.dot(A_atomic_new, Yk_new_in.transpose()).transpose()
        Ya_new_out = np.dot(A_atomic_new, Yk_new_out.transpose()).transpose()

        # Enthalpies
        h_new_in = np.sum(gas_red_mech.partial_molar_enthalpies/gas_red_mech.molecular_weights*Yk_new_in, axis=1)
        h_new_out = np.sum(gas_red_mech.partial_molar_enthalpies/gas_red_mech.molecular_weights*Yk_new_out, axis=1)

        # Residuals
        res_rel_Ya_in = np.abs((Ya_in - Ya_new_in)/Ya_in)
        res_rel_Ya_out = np.abs((Ya_out - Ya_new_out)/Ya_out)
        res_rel_h_in = np.abs((h_in - h_new_in)/h_in)
        res_rel_h_out = np.abs((h_out - h_new_out)/h_out)

        # Checking that residuals are all small
        assert res_rel_Ya_in.max() < 1.0e-10
        assert res_rel_Ya_out.max() < 1.0e-10
        assert res_rel_h_in.max() < 1.0e-10
        assert res_rel_h_out.max() < 1.0e-10


    # Database final processing
    def process_database(self, plot_distributions = False, distribution_species=[]):

        self.is_processed = True

        # Pressure removed by default (can be changed later)
        remove_pressure_X = True
        remove_pressure_Y = True

        self.list_X_p_train = []
        self.list_X_p_val = []
        self.list_Y_p_train = []
        self.list_Y_p_val = []

        for i_cluster in range(self.nb_clusters):
            
            print("")
            print(f"CLUSTER {i_cluster}:")

            # Isolate cluster i
            X_p = self.X[self.X["cluster"]==i_cluster]
            Y_p = self.Y[self.X["cluster"]==i_cluster]

            # Reset indexes
            X_p = X_p.reset_index(drop=True)
            Y_p = Y_p.reset_index(drop=True)

            # Removing useless columns
            X_cols = X_p.columns.to_list()
            list_to_remove = ["Prog_var", "HRR", "cluster"]
            if remove_pressure_X:
                list_to_remove.append('Pressure')
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
            if self.log_transform_X>0 and self.log_transform_Y==0 and self.output_omegas==True:
                X_p_save = X_p.loc[:, X_cols[1:]]


            #Applying transformation (log of BCT)
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
            if self.output_omegas==True:
                if self.log_transform_X>0 and self.log_transform_Y==0:  # Case where X has been logged but not Y
                    Y_p = Y_p.subtract(X_p_save.reset_index(drop=True))
                else:
                    Y_p = Y_p.subtract(X_p.loc[:,X_cols[1:]].reset_index(drop=True))

            # Renaming columns
            X_p.columns = [str(col) + '_X' for col in X_p.columns]
            Y_p.columns = [str(col) + '_Y' for col in Y_p.columns]

            # Train validation split
            X_train, X_val, Y_train, Y_val = train_test_split(X_p, Y_p, test_size=0.25, random_state=42)

            # Forcing constant N2
            n2_cte = True # By default -> To make more general
            if n2_cte:
                if self.output_omegas:
                    Y_train["N2_Y"] = 0.0
                    Y_val["N2_Y"] = 0.0
                else:
                    Y_train["N2_Y"] = X_train["N2_X"]
                    Y_val["N2_Y"] = X_val["N2_X"]

            #Saving in lists
            self.list_X_p_train.append(X_train)
            self.list_X_p_val.append(X_val)
            self.list_Y_p_train.append(Y_train)
            self.list_Y_p_val.append(Y_val)

            # Saving datasets
            print(">> Saving datasets")
            X_train.to_csv(self.dtb_folder + "/" + self.database_name + f"/cluster{i_cluster}/X_train.csv", index=False)
            X_val.to_csv(self.dtb_folder + "/" + self.database_name + f"/cluster{i_cluster}/X_val.csv", index=False)
            Y_train.to_csv(self.dtb_folder + "/" + self.database_name + f"/cluster{i_cluster}/Y_train.csv", index=False)
            Y_val.to_csv(self.dtb_folder + "/" + self.database_name + f"/cluster{i_cluster}/Y_val.csv", index=False)

            # Plotting distributions if necessary
            if plot_distributions:
                print(">> Plotting distributions")
                for spec in distribution_species:
                   self.plot_distributions(X_train, X_val, Y_train, Y_val, spec, i_cluster)

        

    def print_data_size(self):

        # Printing number of elements for each class
        size_total = 0
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


        fig1.savefig(self.dtb_folder + "/" + self.database_name + f"/cluster{i_cluster}/distrib_{species}_train.png", dpi=300, bbox_inches='tight')
        fig2.savefig(self.dtb_folder + "/" + self.database_name + f"/cluster{i_cluster}/distrib_{species}_val.png", dpi=300, bbox_inches='tight')



    def check_inputs(self):

        if self.log_transform_X==0 and self.log_transform_Y>0:
            sys.exit("ERROR: log_transform_Y cannot be active if log_transform_X==0 !")



    # 2-D joint PDF plotting of two variable
    def density_scatter(self, var_x , var_y, sort = True, bins = 100):
        # Functions from https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib

        x = self.X[var_x]
        y = self.X[var_y]

        fig , ax = plt.subplots()
        data , x_e, y_e = np.histogram2d(x, y, bins = bins, density = False )
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



    # Marginal PDF of a given variable in the dataframe
    def plot_pdf_var(self, var): 
            
        # Temperature histogram
        fig, ax = plt.subplots()
        
        sns.histplot(data=self.X, x=var, ax=ax, stat="probability",
                     binwidth=20, kde=True)

        fig.tight_layout()

