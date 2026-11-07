Theory and algorithms
=====================

Models and algorithms used in ARF, with their equations. The [README](../README.md) describes the workflow, the files and the inputs; it links to the sections below.

+ [EMST mixing model (`EMST_POPE`)](#emst-mixing-model-emst_pope): stochastic reactors
+ [Temperature update of the ANN chemistry step](#temperature-update-of-the-ann-chemistry-step)
+ [Renewal of the particles (`residence_time`)](#renewal-of-the-particles-residence_time)
+ [Element conservation of the ANN mass fractions (`yk_renormalization`)](#element-conservation-of-the-ann-mass-fractions-yk_renormalization)
+ [Hybrid ANN/CVODE model (`hybrid_ann_cvode`)](#hybrid-anncvode-model-hybrid_ann_cvode)
+ [Rollout training](#rollout-training)
+ [Element projection in training](#element-projection-in-training)
+ [Loss weighting](#loss-weighting)
+ [Simulation error](#simulation-error)
+ [Speed-up benchmark](#speed-up-benchmark)

# EMST mixing model (`EMST_POPE`)

Model of S. Subramaniam & S.B. Pope, Combust. Flame 115 (1998) 487-514, translated from the Fortran implementation of Z. Ren, S. Subramaniam & S.B. Pope (2002, http://tcg.mae.cornell.edu/emst, `emst.f` and `emst_subs.f`), in `stochastic_reactors_data_gen/emst_pope.py`. It is applied at each time step $dt$ to the active particles (`ParticlesCloud._mix_emst_pope`), over the normalized time $\Delta s = dt / \tau$ ($\tau$: `mixing_time`).

**Variables.** The particles have compositions $f_{ij}$ ($Y_k$, $h_s$ and $Y_{c,u}$ are mixed), weights $w_i$ (`Mass`) and an age property $a_i$. The tree is built on the scaled compositions $g_{ij} = f_{ij} / f_j^{scale}$: as in the example of `emst.f`, $f^{scale} = 0.1$ for the mass fractions, and $h_s$ and $Y_{c,u}$ are not used for the tree ($f^{scale} = 10^{16}$). The variance function $VF = \sum_j \mathrm{var}(g_j)$ (weighted) decays as $\exp(-\Delta s)$ in expectation.

**Age property** (`Particle.emst_age`, internal to the model, not the age `Age` of `solutions.h5`): remaining time of the current state, in units of $\tau$; $a_i > 0$: the particle is mixing, $a_i < 0$: it is not. The durations of the states are uniform in $[0.0178, 0.3157]$ (mixing) and $[0.1666, 0.1667]$ (non-mixing). Initially, a particle is mixing with the probability $p_0 = \langle Z_1 \rangle / (\langle Z_0 \rangle + \langle Z_1 \rangle) \approx 0.5$, with an age drawn from the stationary distribution of the remaining time (density $\propto 1$ below the shortest duration, decreasing linearly to 0 at the longest). After mixing over $\Delta s$, $|a_i|$ decreases by $\Delta s$; a particle whose state ends switches to the other state with a new duration.

**Mixing step** (repeated until $\Delta s$ is mixed, each over at most $0.0178$, the shortest mixing duration):

1. *Tree particles*: the mixing particles, plus non-mixing ones (closest to the end of their state first, which then start mixing) until $p \ge VF / (2.5\, VF_T)$, with $p$ the weight fraction and $VF_T$ the variance function of the tree particles (the ensemble cannot decay more than 2.5 times faster than them).
2. *Tree*: Euclidean minimum spanning tree of the tree particles in the space of $g$ (Prim's algorithm). Edge $\nu$ gets the coefficient $B_\nu = 2 \min(W_\nu, 1 - W_\nu)$, $W_\nu$ the weight fraction of the tree on one side of the edge: edges in the middle of the tree mix more.
3. *Mixing* of the tree particles along the edges:

$$w_i \frac{d\phi_i}{ds} = -\alpha \sum_{\nu \ni i} B_\nu (\phi_i - \phi_{j_\nu})$$

   integrated by an implicit Euler step, $(\mathrm{diag}(w) + \eta L)\, \phi^{new} = \mathrm{diag}(w)\, \phi$, with $L$ the Laplacian of the tree weighted by $B$ and $\eta = \alpha\, \delta s$ (weighted means conserved).
4. *Amount of mixing*: $\eta$ is increased from its estimate $1.08\, \alpha_{est}\, \delta s$, $\alpha_{est} = VF / (2 \sum_\nu B_\nu |\Delta\phi_\nu|^2)$ (initial decay rate), until $VF(\eta) \le VF\, e^{-\delta s}$; then halved until no composition changes by more than 0.3 standard deviations. The time mixed is $\delta s_{done} = \ln(VF / VF(\eta))$; if it exceeds $\delta s$, the mixing is accepted with the probability $\delta s / \delta s_{done}$ (rejection method), so that the expected decay is right.
5. Ages incremented by $\delta s_{done}$.

When the range of the scaled compositions is below $4 \times 10^{-4}$ (or the largest variance below $10^{-12}$), the IEM model is used instead ($g \leftarrow g - (1 - e^{-\Delta s/2})(g - \langle g \rangle)$). Steps smaller than $10^{-6}$ are done by mixing over $2 \times 10^{-6}$, accepted with a probability.

# Temperature update of the ANN chemistry step

The networks predict the mass fractions only; the new temperature follows from the conservation of the mass enthalpy $h = \sum_k h_k(T) Y_k$ at constant pressure (`NNInference.temperature_after`, and `NN_manager._temperature_after` in rollout training). Linearized around $T(t)$:

$$T(t+dt) = T(t) - \frac{\sum_k h_k(T(t)) \left(Y_k(t+dt) - Y_k(t)\right)}{c_p(t)}, \qquad c_p(t) = \sum_k c_{p,k}(T(t))\, Y_k(t)$$

with $h_k$ [J/kg] and $c_{p,k}$ [J/kg/K] from the NASA polynomials of the mechanism, vectorized over the states (`utilities.species_h_cp`). This is first order in $\Delta T = T(t+dt) - T(t)$: $c_p$ is taken at $T(t)$ and with $Y_k(t)$, which is accurate when the temperature changes little over $dt$.

# Renewal of the particles (`residence_time`)

Without renewal the particles form a closed reactor: once all have burnt and mixed, only near-equilibrium states are added to the database. With `residence_time` $\tau_{res}$ (`ParticlesCloud._renew_particles`), the particles form a partially stirred reactor: from `renewal_start_time`, during each $dt$ each active particle returns to the state of its inlet with the probability

$$p = 1 - \exp(-dt / \tau_{res})$$

The residence times then follow an exponential distribution of mean $\tau_{res}$. Fresh gases keep mixing with hot products, so partially reacted states keep being produced, and a few old particles still sample slow chemistry in burnt gases. Particles are not selected by their state.

+ **Database pairs.** Renewal is done after mixing, so the pairs $X \to Y$ of the database, stored around the reaction step, stay pure chemistry.
+ **State of a renewed particle.** $h_s$, $P$, $Y_k$ and $Y_{c,u}$ of its inlet, mass 1; age 0; a new age property is drawn as at initialization with `EMST_POPE`. Its inlet keeps its share of the inflow: a hot inlet acts as a permanent pilot.
+ **Age.** The age $a$ of a particle (`Age` in `solutions.h5`) adds up $dt$ while it is active and is reset to 0 at renewal, also in a closed reactor. With renewal, its distribution tends to an exponential: after the step, the mean and the standard deviation of $a$ tend to about $\tau_{res}$ (for the mean, $dt / (e^{dt/\tau_{res}} - 1) \approx \tau_{res} - dt/2$). `StochDatabase.plot_age_time` plots their evolution.
+ **Choice of $\tau_{res}$.** If it is too short the reactor can blow out, if it is too long the particles are mostly burnt. It is compared to the mixing time and the chemical times.

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

# Rollout training

The network is usually trained on one step: from a true state at $t$, it predicts the state at $t+dt$. In a simulation it is applied on its own predictions, on states slightly different from the database. Rollout training (`rollout: true` in `dtb_processing.yaml`, `NN_manager._rollout_loss`) applies it $n$ times in a row during training, and the gradient goes through the whole chain.

**Database.** From each state of the database, `generate_multi_dt.py` with `time_step_type: rollout` integrates a constant-pressure reactor over $k\,dt$, $k = 1 \dots n$ (`rollout_dt`, `rollout_steps`): chemistry only, without mixing. Integrating from the initial state to $k\,dt$ gives the same trajectory as $k$ successive steps of $dt$.

**Chain.** Noting $\hat{Y}^{(0)} = Y(t)$ and $\hat{T}^{(0)} = T(t)$, at step $k$ the network input is built from $\hat{Y}^{(k-1)}$ and $\hat{T}^{(k-1)}$ exactly as at inference (`NNInference.advance`), and its output gives $\hat{Y}^{(k)}$. The temperature follows from the [conservation of enthalpy](#temperature-update-of-the-ann-chemistry-step), with the same formula as at inference, so that the network sees its own temperature errors:

$$\hat{T}^{(k)} = \hat{T}^{(k-1)} - \frac{\sum_i h_i(\hat{T}^{(k-1)}) \left( \hat{Y}_i^{(k)} - \hat{Y}_i^{(k-1)} \right)}{\sum_i c_{p,i}(\hat{T}^{(k-1)}) \, \hat{Y}_i^{(k-1)}}$$

N2, if not predicted, keeps its value at $t$ (`Y_N2_train`, `Y_N2_val` in `training_data.h5`); it only contributes to $c_p$. No element correction is applied in the chain, unless the networks are trained with the [element projection](#element-projection-in-training).

**Loss.** The loss is on the error of the predicted state, for outputs as mass fractions or as differences. With $f$ the transform of the outputs, $Y_k(t + k\,dt)$ the reference (`Y_rollout_train`, `Y_rollout_val`: transformed, not scaled) and $\sigma$ the standard deviation of each output (`Yscaler`):

$$\mathcal{L} = \frac{1}{n} \sum_{k=1}^{n} \ell \left( \frac{f(\hat{Y}^{(k)}) - f(Y(t + k\,dt))}{\sigma} \right)$$

with $\ell$ the loss function (`loss`: MSE, L1 or Huber). With `output_omegas: true`, the network output is compared to the difference from the *predicted* state, $f(Y(t + k\,dt)) - f(\hat{Y}^{(k-1)})$, with the gradient of $\hat{Y}^{(k-1)}$. The errors accumulated over the previous steps therefore stay in the loss, as with `output_omegas: false`. Comparing it to the true increment $f(Y(t + k\,dt)) - f(Y(t + (k-1)\,dt))$ would only penalize the error made at each step, not its accumulation. For $n=1$, $\mathcal{L}$ is the usual loss. The validation loss, which selects the best model and drives `ReduceLROnPlateau`, is the same rollout loss; the conservation diagnostics of training stay on the first step. After training, the validation loss of each step $k$ with the best model is written in `training/cluster_<i>/rollout_val_loss_per_step.csv` and in the training log: it shows how the error grows along the chain.

**Clusters.** Networks are trained one cluster at a time: a chain uses the network of the cluster of its initial state at all steps, even if the state moves to another cluster, while inference picks the cluster at each step.

**Limitation with Box-Cox.** Rollout is not validated with Box-Cox (`log_transform: 2`); with the logarithm, the inverse $\exp(y)$ is always valid and positive. The inverse Box-Cox is $Y = (\lambda y + 1)^{1/\lambda}$, and nothing keeps the network output above $y = -1/\lambda$, which corresponds to $Y = 0$. Below this limit the base $\lambda y + 1$ is negative:

+ If $1/\lambda$ is not an integer (e.g. $\lambda = 0.3$), $Y$ is NaN.
+ If $1/\lambda$ is an even integer (e.g. the default $\lambda = 0.1$), the result is valid but mirrored around the limit: $y = -11$ behaves like $y = -9$. The gradient of the next input with respect to the output changes sign, and large undershoots give non-physical values ($y = -25$ gives $Y \approx 58$).

The same formula is used at inference, but in rollout training it also enters the gradients and the temperature update. Limiting the base in `transforms.inverse` would have to use a small positive value and not 0: at $Y = 0$ the derivative of $Y^\lambda$ is infinite, and the gradient becomes NaN.

# Element projection in training

With `project_elements: true` in `learning` (`NN_manager._one_step_loss`, `tools/element_projection.py`), the correction `3` of [`yk_renormalization`](#element-conservation-of-the-ann-mass-fractions-yk_renormalization), noted $P$, is used in the training loss. It is not a layer of the network: the network still predicts the state $Y^*$ before correction, so the saved models and inference do not change. With $\ell$ the loss function, on scaled outputs:

$$\mathcal{L} = \ell\big(P(Y^*)\big) + \lambda\, \ell\big(Y^*\big), \qquad \lambda = \texttt{raw\_state\_loss\_weight}$$

$P(Y^*)$ is converted back to a network output (`transforms.Yk_to_output`) to be compared with the usual target, whatever the outputs (mass fractions or differences). The mass fractions are clipped before the transform, as the network inputs: errors below `threshold` do not count, and species clipped in the term $\ell(P(Y^*))$ have no gradient. The gradient goes through the solution of the linear system of $P$. Without N chemistry, N is not considered: it is carried by N2, not predicted.

**Term $\lambda\, \ell(Y^*)$.** The true state conserves the elements, so both terms have the same minimum. With $\lambda = 0$, the loss barely controls the error of $Y^*$ along the directions removed by $P$ (common rescaling of the species containing an element): the conservation error of $Y^*$ can drift. It would make the correction larger and the error proxy of the [hybrid model](#hybrid-anncvode-model-hybrid_ann_cvode) less reliable, as this proxy is computed before correction. A very large $\lambda$ is the usual training, with the correction at inference only.

**Validation.** The validation loss (best model, `ReduceLROnPlateau`, per-step losses of rollouts) is $\ell(P(Y^*))$, the state used at inference. The conservation diagnostics of training (`mass_conservation.png`, `elements_conservation.png`) stay on $Y^*$: they show the drift, not the conservation of the corrected state, which is exact.

**Rollout.** The chain carries the projected state, and the temperature update uses it, as at inference with `yk_renormalization: 3`. Each step has the loss above.

**Inference.** Networks trained this way should be used with `yk_renormalization: 3`; `NNInference` prints a warning otherwise (legitimate for comparisons). The correction factors $1 + \sum_B \mu_B c_{B,k}$ can be far from 1 early in training, where positivity is not guaranteed; negative mass fractions are then clipped by the transform.

# Loss weighting

With `output_omegas: true` and `log_transform_Y: 1`, the target of species $k$ at point $i$ is the increment $\hat\omega_{ik} = \ln Y_k(t+dt) - \ln Y_k(t)$, scaled as $y_{ik} = (\hat\omega_{ik} - \mu_k)/\sigma_k$, with $\mu_k$ and $\sigma_k$ the mean and standard deviation of the cluster. The mean cancels in the error: with $e_{ik}$ the relative error on $\hat\omega_{ik}$, the MSE loss of the point is

$$\sum_k \left(\frac{\hat\omega^{pred}_{ik} - \hat\omega^{ref}_{ik}}{\sigma_k}\right)^2 = \sum_k e_{ik}^2 \left(\frac{\hat\omega^{ref}_{ik}}{\sigma_k}\right)^2 \;\overset{e_{ik} = e_i}{=}\; e_i^2\, \Omega_i^2, \qquad \Omega_i^2 = \sum_k \left(\frac{\hat\omega^{ref}_{ik}}{\sigma_k}\right)^2$$

if all species of the point have the same relative error $e_i$. $\Omega_i$ is the norm of the scaled increments of the point ($\Omega = 0$: no change of the state). For two kinds of states, reacting ($r$, $N_r$ points) and burnt ($b$, $N_b$ points), the ratio of their losses is

$$\frac{\mathcal{L}_b}{\mathcal{L}_r} = \frac{N_b}{N_r}\,\frac{e_b^2}{e_r^2}\,\frac{\Omega_b^2}{\Omega_r^2}$$

For the same relative error, a point of the burnt gases ($\hat\omega \sim 10^{-4}$) counts about $10^6$ times less than a point of the reaction zone ($\hat\omega \sim 10^{-1}$), whatever the numbers of points. A small relative error on the increments of the burnt gases is harmless for one step, but it acts as a bias over thousands of steps, and the state relaxes to a false equilibrium (see the 0D tests, `Temperature_stability.png` and `log_changes/`).

With `loss_weighting_p` $= p > 0$ (`tools/loss_weighting.py`, `NN_manager._loss_weights`), the MSE loss is weighted with one weight per point, the same for all its species:

$$\mathcal{L} = \frac{1}{n\, K}\sum_{i,k} w_i\,(\hat y_{ik} - y_{ik})^2, \qquad w_i = \frac{(\Omega_i^2 + \varepsilon^2)^{-p}}{\big\langle (\Omega^2 + \varepsilon^2)^{-p} \big\rangle_{\text{train}}}$$

with $\varepsilon$ = `loss_weighting_eps` [-], in units of $\sigma_k$. The ratio of the losses becomes, for $\Omega \gg \varepsilon$,

$$\frac{\mathcal{L}_b}{\mathcal{L}_r} = \frac{N_b}{N_r}\,\frac{e_b^2}{e_r^2}\left(\frac{\Omega_b}{\Omega_r}\right)^{2-2p}$$

+ $p = 0$ is the usual loss; $p = 1$ gives the same weight to the same relative error (the balance of the loss is the ratio of the numbers of points); intermediate $p$ tolerate a relative error $e_b/e_r \propto (\Omega_b/\Omega_r)^{p-1}$ in the burnt gases.
+ $\varepsilon$ bounds the weights: their ratio is at most $(\max\Omega/\varepsilon)^{2p}$. Below $\varepsilon$, the error is absolute (equilibrium, solver noise). The quantiles of $\Omega$ are printed at the start of training to choose it.
+ One weight per point, from all species: a species whose increment crosses 0 in the reaction zone (intermediate species at its peak) or is clipped at `threshold` does not make the point look like burnt gases. The balance between the species of a point is the one of the usual loss.
+ The weights are normalized to a mean of 1 on the training points; validation points use the same normalization. $\sigma_k$ are those of the cluster, so $\Omega$ and $\varepsilon$ are relative to the spread of the increments of the cluster.

**Resampling on $\Omega$.** `LearningDatabase.undersample_Omega` keeps the same number of points in each bin of $\log_{10}\Omega$ ($\sigma_k$ of the whole database, $\Omega$ below `Omega_min` in the first bin). The share of the loss of a bin of $\Omega$, for the same relative error, is

$$S_{\text{bin}} \propto N_{\text{bin}}\;\Omega_{\text{bin}}^{\,2-2p}$$

The resampling sets $N_{\text{bin}}$, the weights set the factor $\Omega^{2-2p}$: they are complementary. With the resampling and $p = 1$, each bin of $\Omega$ has the same share of the loss; with $p < 1$, the bins of large increments (reaction zone) have a larger share.

**Density of $\Omega$ in the weights.** Instead of discarding points, `loss_weighting_q` $= q > 0$ multiplies the weights by a density factor:

$$w_i = \frac{(\Omega_i^2 + \varepsilon^2)^{-p}\;\max\big(N_{\text{bin}(i)},\, N_{\min}\big)^{-q}}{\langle\cdot\rangle_{\text{train}}}$$

with $N_{\text{bin}(i)}$ the number of training points of the cluster in the bin of $\log_{10}\max(\Omega_i, \varepsilon)$ of point $i$ (bins of width given by the Freedman-Diaconis rule, as `undersample_1D`), and $N_{\min}$ = `loss_weighting_min_count`. The share of the loss of a bin becomes

$$S_{\text{bin}} \propto N_{\text{bin}}^{\,1-q}\;\Omega_{\text{bin}}^{\,2-2p}$$

$q = 1$ is equivalent, in expectation, to the undersampling on $\Omega$, without discarding points; $0 < q < 1$ is a partial rebalancing (inverse-frequency weighting of imbalanced regression: Yang et al., ICML 2021, https://arxiv.org/abs/2102.09554). The weights of rare bins are large: their few points can be overfitted, and the gradients of the mini-batches are noisier. $N_{\min}$ bounds these weights (bins with fewer points have the weights of a bin of $N_{\min}$ points), and the effective fraction of points shows their concentration. Validation points use the bins and counts of the training points.

The weights move accuracy from the reaction zone to the small increments: the network has a finite capacity, so a large $p$ or a small $\varepsilon$ can degrade the ignition. The training prints the minimum and maximum weights and the effective fraction of points $(\sum w)^2 / (n \sum w^2)$ (1 for equal weights), and logs the unweighted validation loss (`val_loss_unweighted`) to compare models trained with different weights; the best model is selected on the weighted validation loss. The weighting needs the MSE loss (implemented as the MSE of $\sqrt{w}\,\hat y$ and $\sqrt{w}\,y$), and is not available with rollout training (the increments of the following steps depend on the predictions). With `project_elements`, both terms of the loss are weighted. Example: `scripts/H2_IGN_KERNEL_WEIGHTED_LOSS`.

# Simulation error

The simulation error of a 0-D case $i$ run with time step $dt$ is the normalized error of Mehl & Aubagnac (variable time step ANN, eq. 6), following Chen et al. (Energy & Fuels 37, 2023):

$$\mathcal{E}_i(dt) = \frac{1}{N_i} \sum_{j=1}^{N_i} \frac{1}{N_S + 1} \left( \sum_{k=1}^{N_S} \frac{|f(Y_{k,j}^{NN}) - f(Y_{k,j}^{ref})|}{\sigma_k} + \frac{|T_j^{NN} - T_j^{ref}|}{\sigma_T} \right)$$

with $f$ the transform of the network inputs (log or Box-Cox), $N_S$ the number of species of the network (N2 excluded without N chemistry), $\sigma$ the standard deviations of the training inputs over all clusters, and $N_i$ the number of time steps of the simulation. Each 0-D simulation stops when the CVODE temperature is within `T_eq_tolerance` of the equilibrium (HP) temperature of the initial mixture, and the ANN is run for the same number of steps; `max_simulation_time` only caps the simulations that do not reach equilibrium (`equilibrium_reached: false`). $\mathcal{E}(dt)$, the mean of $\mathcal{E}_i$ over the simulations run with $dt$, is written in `simulation_errors_<case type>_by_dt.csv` and plotted in `simulation_errors_<case type>_by_dt.png` (mean with standard deviation bars, and box plot of $\mathcal{E}_i$ per $dt$). The contribution of each variable, $E_T$ (`E_T`) and $E_k$ (`E_<species>`), is the term of $\mathcal{E}_i$ of this variable averaged over the steps; $\mathcal{E}_i$ is their mean. It shows which species drive the error (`simulation_errors_<case type>_by_variable.png`). For 1-D flames, $\mathcal{E}_i$ is a priori: each grid point of the converged flame (in `x_window`) is advanced by one step $dt$ with CVODE and with the ANN, and the sum over $j$ runs over these points instead of the time steps. It measures the error of one step, not its accumulation, so it is smaller than the 0-D error for the same network; it also decreases when the domain contains more points in fresh or burnt gases, hence `x_window`. `simulation_errors_<case type>_by_cluster.csv` gives, for each `dt` and cluster, the number of steps (0-D) or grid points (1-D) advanced by the network of the cluster, the mean and standard deviation of their error (mean over the variables of the normalized differences) and the mean of each variable. Cluster -1 gathers the steps done by CVODE in hybrid mode and the 1-D points below `T_threshold`. In 0-D the error of a step includes the error accumulated over the previous steps, possibly in other clusters; the 1-D errors (one step from the converged flame) are attributed to their cluster exactly.

# Speed-up benchmark

The speed-up benchmark (`speedup_benchmark`) draws states from the validation sets of `training_data.h5` and times `NNInference.advance` on batches of `batch_sizes` states (minimum over 3 calls, clustering not included as the clusters of the validation states are known) against CVODE with a new reactor for each state, as at each time step of a CFD code (time per state measured on `nb_states_cvode` states). `speedup_benchmark.csv` gives, for each batch size, the ANN time [s], the ANN and CVODE times per state [s] and the speed-up.
