from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting

testing_parameters = {}

testing_parameters["models_folder"] = "MODEL_H2_POF_2023"     # Folder for the ML model
testing_parameters["spec_to_plot"] = ["N2", "H2", "O2", "H2O", "H2O2"]          # List of species to plot

testing_parameters["yk_renormalization"] = True

testing_parameters["hybrid_ann_cvode"] = False             # CVODE helping ANN based on a conservation criterion
testing_parameters["hybrid_ann_cvode_tol"] = 5.0e-05      # Tolerance for the CVODE switching

test = NNTesting(testing_parameters)

phi = 0.4
T0 = 1200.0
pressure = 101325.0
dt = 0.5e-6
nb_ite = 1000

test.test_0D_ignition(phi, T0, pressure, dt, nb_ite)