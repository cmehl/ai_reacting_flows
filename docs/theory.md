Theory and algorithms
=====================

Models and algorithms used in ARF, with their equations. The [README](../README.md) describes the workflow, the files and the inputs; it links to the sections below.

+ [EMST mixing model (`EMST_POPE`)](#emst-mixing-model-emst_pope): stochastic reactors
+ [Temperature update of the ANN chemistry step](#temperature-update-of-the-ann-chemistry-step)
+ [Element conservation of the ANN mass fractions (`yk_renormalization`)](#element-conservation-of-the-ann-mass-fractions-yk_renormalization)
+ [Hybrid ANN/CVODE model (`hybrid_ann_cvode`)](#hybrid-anncvode-model-hybrid_ann_cvode)
+ [Simulation error](#simulation-error)
+ [Speed-up benchmark](#speed-up-benchmark)

# EMST mixing model (`EMST_POPE`)

Model of S. Subramaniam & S.B. Pope, Combust. Flame 115 (1998) 487-514, translated from the Fortran implementation of Z. Ren, S. Subramaniam & S.B. Pope (2002, http://tcg.mae.cornell.edu/emst, `emst.f` and `emst_subs.f`), in `stochastic_reactors_data_gen/emst_pope.py`. It is applied at each time step $dt$ to the active particles (`ParticlesCloud._mix_emst_pope`), over the normalized time $\Delta s = dt / \tau$ ($\tau$: `mixing_time`).

**Variables.** The particles have compositions $f_{ij}$ ($Y_k$, $h_s$ and $Y_{c,u}$ are mixed), weights $w_i$ (`Mass`) and an age property $a_i$. The tree is built on the scaled compositions $g_{ij} = f_{ij} / f_j^{scale}$: as in the example of `emst.f`, $f^{scale} = 0.1$ for the mass fractions, and $h_s$ and $Y_{c,u}$ are not used for the tree ($f^{scale} = 10^{16}$). The variance function $VF = \sum_j \mathrm{var}(g_j)$ (weighted) decays as $\exp(-\Delta s)$ in expectation.

**Age property** (`EMST_age` in `solutions.h5`, `Particle.emst_age`): remaining time of the current state, in units of $\tau$; $a_i > 0$: the particle is mixing, $a_i < 0$: it is not. The durations of the states are uniform in $[0.0178, 0.3157]$ (mixing) and $[0.1666, 0.1667]$ (non-mixing). Initially, a particle is mixing with the probability $p_0 = \langle Z_1 \rangle / (\langle Z_0 \rangle + \langle Z_1 \rangle) \approx 0.5$, with an age drawn from the stationary distribution of the remaining time (density $\propto 1$ below the shortest duration, decreasing linearly to 0 at the longest). After mixing over $\Delta s$, $|a_i|$ decreases by $\Delta s$; a particle whose state ends switches to the other state with a new duration.

**Mixing step** (repeated until $\Delta s$ is mixed, each over at most $0.0178$, the shortest mixing duration):

1. *Tree particles*: the mixing particles, plus non-mixing ones (closest to the end of their state first, which then start mixing) until $p \ge VF / (2.5\, VF_T)$, with $p$ the weight fraction and $VF_T$ the variance function of the tree particles (the ensemble cannot decay more than 2.5 times faster than them).
2. *Tree*: Euclidean minimum spanning tree of the tree particles in the space of $g$ (Prim's algorithm). Edge $\nu$ gets the coefficient $B_\nu = 2 \min(W_\nu, 1 - W_\nu)$, $W_\nu$ the weight fraction of the tree on one side of the edge: edges in the middle of the tree mix more.
3. *Mixing* of the tree particles along the edges:

$$w_i \frac{d\phi_i}{ds} = -\alpha \sum_{\nu \ni i} B_\nu (\phi_i - \phi_{j_\nu})$$

   integrated by an implicit Euler step, $(\mathrm{diag}(w) + \eta L)\, \phi^{new} = \mathrm{diag}(w)\, \phi$, with $L$ the Laplacian of the tree weighted by $B$ and $\eta = \alpha\, \delta s$ (weighted means conserved).
4. *Amount of mixing*: $\eta$ is increased from its estimate $1.08\, \alpha_{est}\, \delta s$, $\alpha_{est} = VF / (2 \sum_\nu B_\nu |\Delta\phi_\nu|^2)$ (initial decay rate), until $VF(\eta) \le VF\, e^{-\delta s}$; then halved until no composition changes by more than 0.3 standard deviations. The time mixed is $\delta s_{done} = \ln(VF / VF(\eta))$; if it exceeds $\delta s$, the mixing is accepted with the probability $\delta s / \delta s_{done}$ (rejection method), so that the expected decay is right.
5. Ages incremented by $\delta s_{done}$.

When the range of the scaled compositions is below $4 \times 10^{-4}$ (or the largest variance below $10^{-12}$), the IEM model is used instead ($g \leftarrow g - (1 - e^{-\Delta s/2})(g - \langle g \rangle)$). Steps smaller than $10^{-6}$ are done by mixing over $2 \times 10^{-6}$, accepted with a probability.


# Element conservation of the ANN mass fractions (`yk_renormalization`)

The networks do not conserve the elements exactly: the predicted mass fractions $Y_k^*$ at $t+dt$ are corrected (`NNInference.advance`, before the temperature update, which then uses the corrected $Y_k$). The sum $\sum_k Y_k^*$ before correction is kept for the mass check of the hybrid mode. Species not predicted by the network (N2 without N chemistry) have $Y_k^* = Y_k(t)$.

Notations, for the elements $A \in \{C, H, O, N\}$ and the species $k$ of the mechanism:

- $n_{A,k}$ number of atoms of $A$ in species $k$, $W_A$ and $W_k$ molar masses [kg/kmol], $a_{A,k} = n_{A,k} W_A / W_k$;
- $c_{A,k} = 1$ if species $k$ contains $A$ ($n_{A,k} > 0$), else 0;
- element mass fractions $Y_A = \sum_k a_{A,k} Y_k$, of the state at $t$: $Y_A(t)$, of the prediction: $Y_A^*$;
- imbalance of the prediction $\delta_A = Y_A^* - Y_A(t)$.

All corrections give $Y_A = Y_A(t)$ for the 4 elements, hence $\sum_k Y_k = \sum_A Y_A$ is also conserved (species made of C, H, O, N only).

**`0`**: no correction.

**`1`** (`enforce_elements_balance`): one species $s_A$ per element carries the whole imbalance,

$$Y_{s_A} \leftarrow Y_{s_A} - \frac{\delta_A}{a_{A,s_A}}$$

1. C with CO2 (only for states containing C), H with H2, N with N2: these species contain only one of C, H, N, so the three corrections are independent ($\delta_A$ of the prediction);
2. $Y_O$ is recomputed (CO2 changed it) and O is corrected with O2.

Exact and cheapest, but a minor carrier species takes the whole imbalance: with 1 % random errors on burnt H2/air gases at 1500 K, H2 ($Y_{H_2} \sim 10^{-7}$) becomes negative in about half of the lean states, O2 in about half of the rich ones.

**`2`** (`enforce_elements_balance_wan`): Wan *et al.*, Combust. Flame 220 (2020), Eqs. (7)-(8). The imbalance of $A$ is distributed on all the species containing $A$, in proportion to their mass fractions:

$$Y_k \leftarrow Y_k - \frac{Y_k}{Y_A}\, \delta_A = Y_k \frac{Y_A(t)}{Y_A} \quad \text{for } c_{A,k} = 1$$

i.e. a common rescaling of the species containing $A$ ($Y$ and $Y_A$ of the current corrected state).

1. For C, H, N in turn, rescale the species containing the element by $Y_A(t) / Y_A$ (factor 1 if $Y_A = 0$);
2. species containing several of C, H, N (NH3, CH4, ...) couple these corrections: step 1 is repeated until $|Y_A - Y_A(t)| \le 10^{-12}\, Y_A(t)$ for C, H, N (at most 50 passes; 2 are enough in practice, the coupling is of second order);
3. O is corrected with O2 only, as in the paper ($Y_{O_2} \leftarrow Y_{O_2} - \delta_O / a_{O,O_2}$).

C, H and N corrections are multiplications by positive factors (positivity kept), but the O correction has the drawback of `1` in rich mixtures, where O2 vanishes.

**`3`** (`enforce_elements_balance_wan_all`): extension of `2` to O. Each species is corrected in proportion to its mass fraction, with one multiplier $\mu_B$ per element it contains:

$$Y_k \leftarrow Y_k \Big(1 + \sum_B \mu_B\, c_{B,k}\Big)$$

The element mass fractions are linear in $Y$, so the conservation of the 4 elements gives a linear system for the 4 multipliers ($Y^*$ in $M$):

$$\sum_B M_{AB}\, \mu_B = -\delta_A, \qquad M_{AB} = \sum_k a_{A,k}\, Y_k^*\, c_{B,k}$$

1. build $M$ (4 x 4) and $\delta$ for each state;
2. elements absent from a state or from the mechanism ($M_{AA} = Y_A^* = 0$, e.g. C in H2/air): $\mu_A = 0$ (row and column of $M$ replaced by the identity, $\delta_A = 0$);
3. solve the systems of all states at once (`np.linalg.solve`), then apply the correction.

Exact in one step, without iterations. For a single element, $\mu_A = -\delta_A / Y_A^*$: Eq. (8) of Wan *et al.* Positivity holds if $1 + \sum_B \mu_B c_{B,k} > 0$, which is not guaranteed by the system but holds by far for small prediction errors: with 1 % random errors on burnt H2/air gases (lean to rich, 1000 to 1500 K), the factors stay within 0.95 to 1.05, also in rich mixtures where `2` makes O2 negative in about half of the states.

| Flag | Species carrying the correction | Exact | Positivity | Cost (NH3 mechanism, $10^5$ states, 1 % errors) |
|------|------|------|------|------|
| `1` | CO2, H2, N2, O2 | yes | no (minor carriers) | 0.5 µs/state |
| `2` | all species with C, H, N (rescaling); O2 for O | yes (iterated) | C, H, N yes; O2 no in rich mixtures | 2.5 µs/state |
| `3` | all species, for the 4 elements | yes (one solve) | in practice | 1.8 µs/state |

# Hybrid ANN/CVODE model (`hybrid_ann_cvode`)

C. Mehl & D. Aubagnac-Karkar, Phys. Fluids 35, 067115 (2023), https://doi.org/10.1063/5.0151026. The ANN is called for every state; its prediction $Y_k^*$ at $t+dt$ is rejected, and the state is integrated with CVODE instead, when an error proxy $\mathcal{P}$ exceeds a threshold $\epsilon$ (`hybrid_ann_cvode_tol`). The proxy is the error of the prediction on a conserved quantity, computed before the element correction (`yk_renormalization`), which would otherwise remove it (`use_cvode` and `NNInference.elements_error` in `NN_inference.py`):

+ `0`: no hybrid model, the ANN is always used.
+ `1`: error on the sum of the mass fractions (Eq. 10 of the paper):

$$\mathcal{P} = e_\Sigma = \left| \sum_{k=1}^{N_S} Y_k^* - 1 \right|$$

+ `2`: largest relative error on the element mass fractions (Eq. B1), over the elements $A \in \{C, H, O, N\}$ present in the state at $t$:

$$\mathcal{P} = e_e = \max_A \frac{\left| Y_A^* - Y_A^t \right|}{Y_A^t}, \qquad Y_A = \sum_k \frac{n_{A,k} W_A}{W_k} Y_k$$

As $\sum_k Y_k = \sum_A Y_A$, conserving every element conserves the sum of mass fractions, but not the reverse: $e_e$ detects errors that compensate in $e_\Sigma$. Elements absent at $t$ (e.g. C in H2/air) are not considered; species not predicted by the network (N2 without N chemistry) are unchanged, so their element (N) has no error.

In the paper (H2 ignition in 2-D turbulence), both criteria correlate with the error of the ANN and give the same behaviour, with different thresholds: $\epsilon \approx 2 \times 10^{-4}$ for $e_\Sigma$ and $20 \times 10^{-4}$ for $e_e$, which call CVODE during the ignition only. The paper retains $e_\Sigma$, slightly simpler. The ratio $\alpha$ of CVODE calls reduces the speed-up $A$ of the ANN as $A'/A = 1/(\alpha A + 1)$: the hybrid model is interesting when large ANN errors are localized in space or time. The threshold can be chosen a priori from the errors on the validation states.

**A priori check** (`NNTesting.hybrid_criteria_on_validation_set`, run with `validation_set`, as Fig. 12 of the paper). Each validation state of `training_data.h5` is advanced by one ANN step (with the element correction `yk_renormalization`, as in a simulation) and compared to its target in the database. The error of the step is the mean over the variables of the normalized differences, as the simulation error:

$$E = \frac{1}{N_S + 1} \left( \sum_{k=1}^{N_S} \frac{|f(Y_k^{NN}) - f(Y_k^{ref})|}{\sigma_k} + \frac{|T^{NN} - T^{ref}|}{\sigma_T} \right)$$

with $f$ the transform of the network inputs and $\sigma$ the standard deviations of the training inputs. The database targets contain only the species: $T^{ref}$ and $T^{NN}$ at $t+dt$ both follow from the enthalpy conservation from the state at $t$ (`NNInference.temperature_after`). $e_\Sigma$ and $e_e$ of the prediction (before element correction) are plotted against $E$ in `plots_validation/hybrid_e_sum.png` and `hybrid_e_elements.png`, colored by the temperature at $t$. The plot of the selected criterion shows `hybrid_ann_cvode_tol` and the fraction of states above it, i.e. the fraction $\alpha$ of CVODE calls expected on states like those of the database. A criterion is useful if large $E$ go with large proxies.

# Simulation error

The simulation error of a 0-D case $i$ run with time step $dt$ is the normalized error of Mehl & Aubagnac (variable time step ANN, eq. 6), following Chen et al. (Energy & Fuels 37, 2023):

$$\mathcal{E}_i(dt) = \frac{1}{N_i} \sum_{j=1}^{N_i} \frac{1}{N_S + 1} \left( \sum_{k=1}^{N_S} \frac{|f(Y_{k,j}^{NN}) - f(Y_{k,j}^{ref})|}{\sigma_k} + \frac{|T_j^{NN} - T_j^{ref}|}{\sigma_T} \right)$$

with $f$ the transform of the network inputs (log or Box-Cox), $N_S$ the number of species of the network (N2 excluded without N chemistry), $\sigma$ the standard deviations of the training inputs over all clusters, and $N_i$ the number of time steps of the simulation. Each 0-D simulation stops when the CVODE temperature is within `T_eq_tolerance` of the equilibrium (HP) temperature of the initial mixture, and the ANN is run for the same number of steps; `max_simulation_time` only caps the simulations that do not reach equilibrium (`equilibrium_reached: false`). $\mathcal{E}(dt)$, the mean of $\mathcal{E}_i$ over the simulations run with $dt$, is written in `simulation_errors_<case type>_by_dt.csv` and plotted in `simulation_errors_<case type>_by_dt.png` (mean with standard deviation bars, and box plot of $\mathcal{E}_i$ per $dt$). The contribution of each variable, $E_T$ (`E_T`) and $E_k$ (`E_<species>`), is the term of $\mathcal{E}_i$ of this variable averaged over the steps; $\mathcal{E}_i$ is their mean. It shows which species drive the error (`simulation_errors_<case type>_by_variable.png`). For 1-D flames, $\mathcal{E}_i$ is a priori: each grid point of the converged flame (in `x_window`) is advanced by one step $dt$ with CVODE and with the ANN, and the sum over $j$ runs over these points instead of the time steps. It measures the error of one step, not its accumulation, so it is smaller than the 0-D error for the same network; it also decreases when the domain contains more points in fresh or burnt gases, hence `x_window`. `simulation_errors_<case type>_by_cluster.csv` gives, for each `dt` and cluster, the number of steps (0-D) or grid points (1-D) advanced by the network of the cluster, the mean and standard deviation of their error (mean over the variables of the normalized differences) and the mean of each variable. Cluster -1 gathers the steps done by CVODE in hybrid mode and the 1-D points below `T_threshold`. In 0-D the error of a step includes the error accumulated over the previous steps, possibly in other clusters; the 1-D errors (one step from the converged flame) are attributed to their cluster exactly.

# Speed-up benchmark

The speed-up benchmark (`speedup_benchmark`) draws states from the validation sets of `training_data.h5` and times `NNInference.advance` on batches of `batch_sizes` states (minimum over 3 calls, clustering not included as the clusters of the validation states are known) against CVODE with a new reactor for each state, as at each time step of a CFD code (time per state measured on `nb_states_cvode` states). `speedup_benchmark.csv` gives, for each batch size, the ANN time [s], the ANN and CVODE times per state [s] and the speed-up.
