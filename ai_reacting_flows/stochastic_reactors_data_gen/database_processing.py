import os
import shutil
import pickle
import shelve
import random

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class LearningDatabase(object):

    def __init__(self, dtb_processing_parameters):

        # Input parameters
        self.dtb_folder  = dtb_processing_parameters["dtb_folder"]
        self.database_name = dtb_processing_parameters["database_name"]
        self.log_transform  = dtb_processing_parameters["log_transform"]
        self.threshold  = dtb_processing_parameters["threshold"]
        self.output_omegas  = dtb_processing_parameters["output_omegas"]

        # Load database
        self.X = pd.read_csv(self.dtb_folder + "/X_dtb.csv", delimiter=";")
        self.Y = pd.read_csv(self.dtb_folder + "/Y_dtb.csv", delimiter=";")

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


    def apply_temperature_threshold(self, T_threshold):

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




    def process_database(self):

        self.is_processed = True

        # Pressure removed by default (can be changed later)
        remove_pressure_X = True
        remove_pressure_Y = True

        self.list_X_p_train = []
        self.list_X_p_val = []
        self.list_Y_p_train = []
        self.list_Y_p_val = []

        for i_cluster in range(self.nb_clusters):

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
            X_train.to_csv(self.dtb_folder + "/" + self.database_name + f"/cluster{i_cluster}/X_train.csv", index=False)
            X_val.to_csv(self.dtb_folder + "/" + self.database_name + f"/cluster{i_cluster}/X_val.csv", index=False)
            Y_train.to_csv(self.dtb_folder + "/" + self.database_name + f"/cluster{i_cluster}/Y_train.csv", index=False)
            Y_val.to_csv(self.dtb_folder + "/" + self.database_name + f"/cluster{i_cluster}/Y_val.csv", index=False)

        

    def print_data_size(self):

        # Printing number of elements for each class
        size_total = 0
        for i in range(self.nb_clusters):

            size_cluster_i = self.X[self.X["cluster"]==i].shape[0]
            size_total += size_cluster_i

            print(f">> There are {size_cluster_i} points in cluster{i}")

        print(f"\n => There are {size_total} points overall")


