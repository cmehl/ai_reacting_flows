MPI parallelism
===============

How the work is shared between MPI ranks in ARF, and how the ranks exchange data. The [README](../README.md) describes how to launch the runs (`mpirun -n X python ...`).

MPI is used in the three database generation steps:

+ [Stochastic reactors](#stochastic-reactors) (`particles_cloud.py`): chemistry and derived variables of the particles
+ [Canonical flames](#canonical-flames) (`database_flmts.py`): simulations of 0-D reactors and 1-D flames
+ [Multi-dt generation](#multi-dt-generation) (`generate_var_dt_dtb.py`): re-integration of the database states with several $dt$

Training and testing do not use MPI.

**Two kinds of mpi4py calls.** Lowercase methods (`comm.scatter`, `comm.gather`, `comm.allgather`, `comm.bcast`) send any Python object: it is *pickled* (serialized into bytes) by the sender and rebuilt by the receiver, which is costly for many objects. Capitalized methods (`comm.Bcast`, `comm.Allgatherv`, ...) send NumPy arrays: memory is copied directly, without pickling. The stochastic reactors use the capitalized ones, as they exchange particle states at every iteration.

# Stochastic reactors

## Principle

+ **Each rank keeps its own copy of all the particles for the whole run.** All ranks build the same particles at initialization (same inputs; the ages of `EMST_POPE` are drawn with the same seed on all ranks). The attributes that never change (species names, molar masses, atomic matrices, inlet number, ...) are therefore never exchanged.
+ **The chemistry and the derived variables are distributed**: each rank computes its share of the particles. The derived variables are the equivalence ratio, the progress variable (which needs an equilibrium computation, the most expensive part), the mixture fraction and the heat release rate.
+ **Mixing and writing are done by rank 0 only.** The mixing pairs of `CURL` and `CURL_MODIFIED` are also drawn by rank 0 only.
+ **After each step, all ranks get the states of all particles**: only the attributes that change during the simulation are exchanged, packed as an array of floats with one row per particle.

Exchanged attributes (`EXCHANGED_SCALARS` and `EXCHANGED_ARRAYS`, top of `particles_cloud.py`), for example with the H2 mechanism (9 species):

| Scalars (13) | `T`, `P`, `hs`, `Hs`, `mass`, `Yc_u`, `Yc_eq`, `recompute_Yc_eq` (flag, sent as 0.0 or 1.0), `prog_var`, `equiv_ratio`, `mix_frac`, `hrr`, `emst_age` |
|---|---|
| Arrays | `Y` (9), `X` (9), `mass_k` (9), `state` (2 + 9: $h_s$, $P$, $Y_k$) |

One row per particle: 13 + 9 + 9 + 9 + 11 = 51 floats.

**Warning:** a particle attribute that starts changing during the simulation (e.g. in a new mixing model) must be added to `EXCHANGED_SCALARS` or `EXCHANGED_ARRAYS`. Otherwise the other ranks keep its old value, without any error. Test 19 (1 against 2 ranks) only detects it if the attribute changes what is written in `solutions.h5`. Attributes used only by rank 0 need not be exchanged, as the particles of rank 0 are never replaced: e.g. `Le_k` of `CURL_MODIFIED_DD`, used in mixing.

## One iteration (`advance_time`)

```
                                                       particles of the ranks
rank 0 writes the states in solutions.h5 (and X)       identical
_apply_reactions: each rank its share, _gather_states  identical
_update_vars:     each rank its share, _gather_states  identical
rank 0 writes the Y states (build_ml_dtb)
rank 0 mixes (_apply_diffusion)                        rank 0 differs
_share_states_of_root                                  identical
_update_vars:     each rank its share, _gather_states  identical
statistics and termination check, on every rank
```

The statistics and the termination check (`calc_statistics`, `check_termination`: standard deviation of the temperature) are computed by every rank on its own particles. As the particles of all ranks are identical, all ranks take the same decision and stop together; ranks with different particles could stop at different iterations, and the others would wait forever in the next exchange.

## Routines (`particles_cloud.py`, section *MPI exchanges of particle states*)

Example used below: **10 particles, 3 ranks**.

### `_particle_chunk(rank)`: particles of a rank

```python
return np.arange(rank, self.nb_parts_tot, self.size)
```

The particles are dealt to the ranks in turn (round-robin): rank $r$ of $N$ takes particles $r$, $r+N$, $r+2N$, ...

| Rank | Particles |
|---|---|
| 0 | 0, 3, 6, 9 |
| 1 | 1, 4, 7 |
| 2 | 2, 5, 8 |

The particles of an inlet are consecutive in the list. Dealt in turn, they are spread over all ranks: e.g. the hot inlet of an ignition case, whose particles are the only ones reacting at first (chemistry is only computed above `T_threshold`), is shared by all ranks instead of being on the last ones. With contiguous blocks (0-3, 4-6, 7-9), the last ranks would do all the chemistry while the others wait. Each rank computes the split itself (from its rank and the number of ranks): it is never sent.

### `_pack_states(particles)`: particles to array

One row per particle, the exchanged attributes in a fixed order: the 13 scalars, then `Y`, `X`, `mass_k`, `state`. For the 3 particles of rank 1: an array (3, 51).

### `_unpack_states(particles, states)`: array to particles

The reverse: each row is read in the same order and written in the attributes of its particle (`setattr`). The flag `recompute_Yc_eq` is turned back into a boolean, and the arrays are copied, so that a particle does not share memory with the received array.

### `_gather_states(chunk)`: after a distributed step, all ranks get all particles

Called at the end of `_apply_reactions` and `_update_vars`, when each rank has just updated its own particles.

1. **Pack its own particles**: rank 1 packs particles 1, 4, 7 in `local`, an array (3, 51).
2. **Sizes of the blocks of all ranks**, computed by every rank from `_particle_chunk`:
   + `chunks` = `[[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]`;
   + `counts` = `[4·51, 3·51, 3·51]` = `[204, 153, 153]`: number of floats sent by each rank;
   + `displacements` = `[0, 204, 357]`: position of the block of each rank in the received array.
3. **Exchange**: `comm.Allgatherv(local, [states, counts, displacements, MPI.DOUBLE])`. Each rank sends its block, and all ranks receive all blocks in `states`, an array (10, 51), **stacked by rank**. The *v* of `Allgatherv` allows blocks of different sizes (here 4, 3 and 3 particles).

   | Row of `states` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | Particle | 0 | 3 | 6 | 9 | 1 | 4 | 7 | 2 | 5 | 8 |
   | Sent by rank | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 2 | 2 | 2 |

4. **Row of each particle**:
   + `particle_of_row = np.concatenate(chunks)` = `[0, 3, 6, 9, 1, 4, 7, 2, 5, 8]`: particle of each row (second line of the table);
   + `row_of_particle[particle_of_row] = np.arange(10)` gives the reverse, the row of each particle: `[0, 4, 7, 1, 5, 8, 2, 6, 9, 3]` (e.g. particle 4 is in row 5).
5. **Update of the particles of the other ranks**: on rank 1, `others` = `[0, 2, 3, 5, 6, 8, 9]` (its own particles 1, 4, 7 are already up to date), read in rows `row_of_particle[others]` = `[0, 7, 1, 8, 2, 9, 3]` of `states`, and unpacked.

A wrong row mapping (step 4) would give each particle the state of another one, without any error: test 19 detects it.

### `_share_states_of_root()`: after mixing, rank 0 sends all particles

Mixing is done by rank 0 only: afterwards, only its particles are up to date.

1. Every rank computes the number of exchanged values per particle (51) from its own first particle: it is the same on all ranks, so it is not sent.
2. Rank 0 packs its 10 particles in `states` (10, 51); the other ranks allocate an empty array of the same shape.
3. `comm.Bcast(states, root=0)` copies the array of rank 0 to all ranks.
4. The ranks other than 0 unpack it into their particles.

With a single rank, `_gather_states` and `_share_states_of_root` return immediately: there is nothing to exchange.

## Cost

Measured on the H2 ignition case (`scripts/H2_IGN_KERNEL` inputs: 600 particles, `EMST_POPE`), on a workstation, per iteration:

| Version | 1 rank | 4 ranks | 8 ranks |
|---|---|---|---|
| Particles exchanged as pickled objects (`scatter`, `allgather`) | 441 ms | 291 ms | 340 ms |
| Arrays (`Allgatherv`, `Bcast`), contiguous blocks | 112 ms | 71 ms | 114 ms |
| Arrays, round-robin | 118 ms | 63 ms | 99 ms |

Pickling cost even on one rank (mpi4py serializes the objects anyway): a particle is 1.9 kB pickled, mostly constants, against 51 floats (0.4 kB) exchanged now. With more ranks, the time left in the exchanges is mostly waiting: for the slowest rank in the chemistry, and for rank 0, which mixes and writes alone.

# Canonical flames

`database_flmts.py` (class `DatabaseFlamelets`):

+ **Simulations** (0-D reactors, 1-D premixed and diffusion flames): the training simulations are dealt to the ranks in turn (`rows[self.rank::self.size]`). Each rank runs its simulations; the results (pandas DataFrames, pickled) are gathered on rank 0 (`comm.gather`), which concatenates them, sorts them by simulation number and plots them. The full DataFrame is then sent back to all ranks (`comm.bcast`).
+ **Design of experiments**: rank 0 writes the conditions (`sim_train_*.csv`, `sim_test_*.csv`) and the plots; the other ranks wait at `comm.Barrier()`.
+ **Shuffle of the database**: the permutation is drawn by rank 0 and sent to all ranks (`comm.bcast`), so that all ranks apply the same one.
+ **Integration over $dt$** (`_advance_dt_cfd`): the states are dealt to the ranks in turn; each rank integrates its states, and the results and their indices are gathered on all ranks (`comm.allgather`), then put back in the order of the states.
+ **Augmentation** (`augment_dataset`): states dealt in turn, gathered on rank 0 (`comm.gather`) and sent to all ranks (`comm.bcast`).

These exchanges happen once per step of the generation, not at each time step: pickling is not an issue.

# Multi-dt generation

`generate_var_dt_dtb.py` (`GenerateVariable_dt`):

+ **Split**: rank 0 reads the database and sends a contiguous block of states to each rank (`comm.scatter`, once).
+ **Time steps**: each rank draws its own random $dt$ (seed `1234 + rank`), and integrates its states over each of them.
+ **Writing**: gathering all results on rank 0 is not possible for large databases (mpi4py messages are limited to 2 GB). Each rank therefore writes its own group `ITERATION_<rank>` in the h5 file, in turn: rank 0 creates the file, and each rank waits for a message (token) from the previous rank before writing, then sends it to the next one (`comm.recv`, `comm.send`). The file has one group per rank.

# Tests

+ `DummyComm.py` replaces MPI in the tests run on a single process (methods of a communicator with one rank).
+ Test 19 runs the stochastic reactors (`CURL_MODIFIED`, with particles activated later) on 1 rank, then on 2 ranks with `mpiexec`, and compares the two `solutions.h5`. The tolerance (relative $10^{-5}$) covers the differences due to the Cantera gas object of each rank, whose history differs. The heat release rate is compared with an absolute tolerance ($10^{-5}$ times its largest value): for nearly unreacted particles, it is a sum of terms that cancel, without relative accuracy. It also runs the multi-dt generation on 2 ranks. Skipped without `mpiexec`.
