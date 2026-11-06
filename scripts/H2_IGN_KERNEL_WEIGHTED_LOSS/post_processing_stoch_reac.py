# # Stochastic reactors database post-analysis
#
# Some functions have been implemented to visualize generated stochastic reactors databases.
#
# Additional functions may be added by coding them directly in the file *stochastic_reactors_data_gen/post_processing*.

from ai_reacting_flows.stochastic_reactors_data_gen.post_processing import StochDatabase

# We first define the database to be processed and the folder where the plots will be stored:
stoch_dtb_folder = "./STOCH_DTB_H2_POF_2023"
save_folder = "./postpro_dtb"

# The database is then read in a *StochDatabase* object which contains the methods to perform the post-processing:
stoch_dtb = StochDatabase(stoch_dtb_folder, save_folder, with_traj=False)

# # Function to scatter plot Y_k=f(T) for all time steps on a list on specified species:
# # stoch_dtb.compute_1D_premixed(0.4, 101325.0, 300.0, "H2", "../data/chemical_mechanisms/mech_H2.yaml", "Mix")
# stoch_dtb.plot_T_Yk(["H2", "H", "H2O", "O2"])

# # A generic function is available to scatter plot *var_x* as a function of *var_y*, colored by *var_c*:
# var_x = "Temperature"
# var_y = "H2"
# var_c = "Prog_var"
# stoch_dtb.plot_generic(var_x, var_y, var_c)

# # Some functions are also available to plot individual iterations, here is an example for $(T,Y_k)$ plots:
# stoch_dtb.plot_T_Yk_indiv(["H2", "H"], iteration=3)

# # Function to plot $T_{mean}=f(t)$ for each inlet:
# stoch_dtb.plot_traj_T_time()

# # Function to plot the evolution of some particle variables (here $H_2$) over time for a given inlet:
# nb_inlet = 2
# stoch_dtb.plot_indiv_traj(nb_inlet, "H2")


# # A function to plot the density of points is also available:
# stoch_dtb.density_scatter("Temperature" , "H2", sort = True, bins = 20)

stoch_dtb.plot_pdf_all("Prog_var")
stoch_dtb.plot_pdf_all("Temperature")
stoch_dtb.plot_pdf_all("HRR")
stoch_dtb.plot_pdf_all("log_abs_HRR")

# Perform animation
var1 = "Temperature"
var2 = "H2"
var_color = "Prog_var"
stoch_dtb.plot_animation(var1, var2, var_color, iterations=None, step=100, interval=300, save_path=None, fps=5)