[![DOI](https://zenodo.org/badge/531591043.svg)](https://zenodo.org/badge/latestdoi/531591043)

AI FOR REACTING FLOWS (ARF)
===========================

The project *ai_for_reacting_flows* is a tool enabling to replace complex chemical kinetics calculations in a CFD code by a machine learning model. The idea is that typical exact solvers used to resolve such problems (such as [CVODE](https://computing.llnl.gov/projects/sundials/cvode)) are very expensive. Using fitted models, such as Artificial Neural Networks (ANN), enables to speed-up calculations by a significant factor. 

This package features the following functionalities:

+ A module to build database based on the idea of stochastic reactors

+ A module to generate machine learning models to advance the chemical state of a mixture

This package is made by a very ugly man.

# Package installation

In the current version of the code, Python 3.9.5 is used. The installation of the Python libraries necessary for this project can be performed using pip and the *requirements.txt* file:

```
pip install -r requirements.txt
```

The project itself must be installed as a package, which can be done by going in the root folder and using the following command:

```
python -m pip install -e .
```

Additional details on how to install a home-made package can be found [here](https://realpython.com/python-import/#create-and-install-a-local-package).

In order to run the database generation, an MPI distribution is also necessary. In the current version of the code, the preconized version is OpenMPI/4.1.1.


# Database generation

The *ai_reacting_flows* package includes routines to generate databases on which the model training is subsequently performed. 

The current version of the code only includes the generation of databases for combustion applications. It is based on the methodology used in the paper of [Wan *et al.*](https://www.sciencedirect.com/science/article/pii/S0010218020302170), which is based on the computation of 0-D stochastic particles. Each particle represents a chemical state, and the mixing of particles is aimed at mimicking diffusion taking place in actual systems. 

The following steps must be performed in practice to generate the database using this model:

1. A file specifying the initial state of particles must be created. This is done using the notebook *generate_inlets_file.ipynb*. 

2. Generation of the stochastic particles dataset. The particles are evolved in time and the raw states are stored. The entire set of encountered states is stored in *databases_states.csv*. For ML purposes, the files *X_dtb.csv* and *Y_dtb.csv* are also created and contain the raw temperature and mass fractions at time $t$ and corresponding $t+dt$, respectively. The parameters for the dataset generation must be specified in the script *generate_stoch_dtb.py* found in the *scripts* folder. The script may be launched on multiple processors using MPI: 

```
mpirun -n nb_procs python generate_stoch_dtb.py
```

3. The raw ML database, composed of  *X_dtb.csv* and *Y_dtb.csv*, must be processed. This includes an eventual transformation of the data (log, Box-Cox, etc...), the possible use of differences $Y_k(t+dt)-Y_k(t)$ as ANN outputs instead of $Y_k(t+dt)$, the clustering of the dataset and the splitting into training/validation datasets. Note that the standardization is performed later, at the ANN model generation stage. This step generates folders *cluster{i}*, $i=1..N_c$ where $N_c$ is the number of data clusters. In each folder, the files *X_train.csv*, *X_val.csv*, *Y_train.csv* and *Y_val.csv* are stored. The processing may be launched using the notebook *generate_ML_dtb.ipynb*, where all input parameters are specified.

Additionaly, a module for post-processing and analyzing the stochastic particles database is provided. An exemple of use is presented in notebook *post_processing_stoch_reac.ipynb*. If necessary, additional functions may be coded in *ai_reacting_flows/stochastic_reactors_data_gen/post_processing.py*.


# ANN model generation

The *ai_reacting_flows* package considers the use of ANN as surrogates for chemistry computation. In particular, the final aim is to find best ANN parameters $p$ such that $Y_k(t+dt)=ANN_p(Y_k(t))$. However, alternative techniques such as random forest could be easily integrated in the code if needed.

An example of model generation may be found in the notebook *ann_model_learning.ipynb*. 


# Simple model testing

Some functions have been implemented to give the ability to the user to perform ANN model evaluation on simple setups. The tests are defined in the class *ModelTesting*, and other configurations might be added in the class if necessary. The tests currently available are the following:

+ Combustion: 0-D homogeneous mixture ignition
+ Combustion: 1-D premixed flame reaction rate

An example on how to read a model and test it on these simple setups is provided in notebook *ann_model_testing.ipynb*.


