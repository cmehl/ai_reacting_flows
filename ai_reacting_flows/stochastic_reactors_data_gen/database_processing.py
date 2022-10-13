import os, sys
import shutil
import pickle
import shelve
import random

import joblib
import pandas as pd
import numpy as np
import h5py
import cantera as ct
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import ai_reacting_flows.tools.utilities as utils


class LearningDatabase(object):

    def __init__(self, dtb_processing_parameters):

        # Input parameters
        self.dtb_folder  = dtb_processing_parameters["dtb_folder"]
        self.database_name = dtb_processing_parameters["database_name"]
        self.log_transform  = dtb_processing_parameters["log_transform"]
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
        self.species_names = self.X.columns[2:-1]
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
        shelfFile["log_transform"] = self.log_transform
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

    
    def get_database_from_h5(self):

        # Solution 0
        h5file_r = h5py.File(self.dtb_folder + "/solutions.h5", 'r')
        data_X = h5file_r.get("ITERATION_00000/X")[()]
        col_names_X = h5file_r["ITERATION_00000/X"].attrs["cols"]
        data_Y = h5file_r.get("ITERATION_00000/Y")[()]
        col_names_Y = h5file_r["ITERATION_00000/Y"].attrs["cols"]

        self.X = pd.DataFrame(data=data_X, columns=col_names_X)
        self.Y = pd.DataFrame(data=data_Y, columns=col_names_Y)

        # Loop on other solutions
        for i in range(1,self.nb_solutions):

            data_X = h5file_r.get(f"ITERATION_{i:05d}/X")[()]
            data_Y = h5file_r.get(f"ITERATION_{i:05d}/Y")[()]

            df_X_current = pd.DataFrame(data=data_X, columns=col_names_X)
            self.X = pd.concat([self.X, df_X_current], ignore_index=True)

            df_Y_current = pd.DataFrame(data=data_Y, columns=col_names_Y)
            self.Y = pd.concat([self.Y, df_Y_current], ignore_index=True)

        h5file_r.close()


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

                for index, row in self.X.iterrows():
                    c = row["Prog_var"]
                    if (c>=c_bounds[i]) & (c<c_bounds[i+1]):
                        row["cluster"] = i
                

        elif clusterization_method=="kmeans":

            # Get states only (temperature and Yk's)
            cols = [0] + [2+i for i in range(self.nb_species)]
            data = self.X.values[:,cols]

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
        to_keep = ["Temperature", "Pressure"] + species_subset + ["Prog_var", "cluster"]
        self.X = self.X[to_keep]
        to_keep = ["Temperature", "Pressure"] + species_subset + ["Prog_var"]
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

        # Moving progress variable and cluster at the end
        self.X = self.X[[c for c in self.X if c not in ['Prog_var', 'cluster']] + ['Prog_var', 'cluster']]
        self.Y = self.Y[[c for c in self.Y if c not in ['Prog_var',]] + ['Prog_var']]

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
            list_to_remove = ["Prog_var", "cluster"]
            if remove_pressure_X:
                list_to_remove.append('Pressure')
            #
            [X_cols.remove(elt) for elt in list_to_remove]   
            X_p = X_p[X_cols] 
            #
            Y_cols = Y_p.columns.to_list()
            list_to_remove = ["Temperature", "Prog_var"]
            if remove_pressure_Y:
                list_to_remove.append('Pressure')
            #
            [Y_cols.remove(elt) for elt in list_to_remove]   
            Y_p = Y_p[Y_cols] 


            # Clip if logarithm transformation
            if self.log_transform>0:
                X_p[X_p < self.threshold] = self.threshold
                #
                Y_p[Y_p < self.threshold] = self.threshold


            #Applying transformation (log of BCT)
            if self.log_transform==1:
                X_p.loc[:, X_cols[1:]] = np.log(X_p[X_cols[1:]])
                Y_p.loc[:, Y_cols] = np.log(Y_p[Y_cols])
            elif self.log_transform==2:
                X_p.loc[:, X_cols[1:]] = (X_p[X_cols[1:]]**self.lambda_bct - 1.0)/self.lambda_bct
                Y_p.loc[:, Y_cols] = (Y_p[Y_cols]**self.lambda_bct - 1.0)/self.lambda_bct


            # If differences are considered
            if self.output_omegas==True:
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

