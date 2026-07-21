from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting

testing_parameters = {}

testing_parameters["models_folder"] = "MODEL_NH3_PYRO_EMST_2"     # Folder for the ML model
testing_parameters["spec_to_plot"] = ["NH3", "H2"]          # List of species to plot

testing_parameters["yk_renormalization"] = False

testing_parameters["hybrid_ann_cvode"] = False             # CVODE helping ANN based on a conservation criterion
testing_parameters["hybrid_ann_cvode_tol"] = 5.0e-05      # Tolerance for the CVODE switching

test = NNTesting(testing_parameters)

T0 = 1200.0
pressure = 101325.0
dt = 5.0e-4
nb_ite = 2000

test.test_0D_pyrolysis(T0, pressure, dt, nb_ite)