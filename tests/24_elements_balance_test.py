"""
Test 24: element corrections of the ANN mass fractions (NNInference, yk_renormalization 1, 2 and 3).

Mass fractions at t+dt are perturbed (relative noise, as an ANN error) from a burnt state.
Check:
- the corrections (one species per element, Wan et al. 2020, its extension to O) give back the C, H, O, N mass
  fractions of the state at t, also for species containing several of C, H, N (NH3, CH4 mechanisms);
- without such species (H2 mechanism), the Wan correction rescales all H-containing species by
  Y_H(t) / Y_H(t+dt) (Eq. (8) of the paper), N2 back to its value at t, and only O2 for O;
- extended to O, the correction keeps positive mass fractions in a rich mixture (little O2), where the O
  correction with O2 only (Wan et al.) makes O2 negative, and it reduces to Eq. (8) for a single element.

Also the conservation errors of the hybrid ANN/CVODE model (Mehl & Aubagnac-Karkar 2023): elements_error is
the largest relative change of the element mass fractions (0 for a prediction conserving the elements), and
use_cvode selects the states from the sum of mass fractions (1) or the element error (2), none for 0.

Independent of the other tests.
"""
import os
import numpy as np
import cantera as ct
import pytest
import ai_reacting_flows.tools.utilities as utils
from ai_reacting_flows.ann_model_generation.NN_inference import NNInference, use_cvode

mech_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms")


def element_corrector(mech):
    """NNInference with only the chemistry needed by the element corrections (no model)."""

    inference = NNInference.__new__(NNInference)
    inference.gas = ct.Solution(os.path.join(mech_folder, mech))
    inference.atomic_array = utils.get_atomic_array(inference.gas)
    inference.W_atoms = utils.get_atomic_weights()

    return inference


def perturbed_states(gas, fuel, n=20, phi=0.8):
    """Burnt states Y_old (n, n_species) at equivalence ratio phi and predictions Y_new with 1 % relative noise."""

    gas.TP = 1500.0, 101325.0
    gas.set_equivalence_ratio(phi, fuel, {"O2": 1.0, "N2": 3.76})
    gas.equilibrate("HP")

    Y_old = np.tile(gas.Y, (n, 1))
    Y_new = Y_old * (1.0 + 0.01 * np.random.default_rng(0).standard_normal(Y_old.shape))

    return Y_old, Y_new


def elements(inference, Y):
    return inference.W_atoms * ((Y / inference.gas.molecular_weights) @ inference.atomic_array.T)


@pytest.mark.parametrize("mech, fuel", [("mech_H2.yaml", "H2"), ("mech_nh3_noArHe.yaml", "NH3"), ("mech_CH4_gri.yaml", "CH4")])
@pytest.mark.parametrize("method", ["enforce_elements_balance", "enforce_elements_balance_wan", "enforce_elements_balance_wan_all"])
def test_elements_conserved(mech, fuel, method):
    inference = element_corrector(mech)
    Y_old, Y_new = perturbed_states(inference.gas, fuel)

    Y_corr = getattr(inference, method)(Y_old, Y_new)

    assert np.allclose(elements(inference, Y_corr), elements(inference, Y_old), rtol=1.0e-10, atol=1.0e-14)


def test_wan_rescaling_H2():
    inference = element_corrector("mech_H2.yaml")
    Y_old, Y_new = perturbed_states(inference.gas, "H2")

    Y_corr = inference.enforce_elements_balance_wan(Y_old, Y_new)

    # H-containing species rescaled by Y_H(t) / Y_H(t+dt), Y_H of the prediction
    ratio_H = elements(inference, Y_old)[:, 1] / elements(inference, Y_new)[:, 1]
    has_H = inference.atomic_array[1] > 0
    assert np.allclose(Y_corr[:, has_H], Y_new[:, has_H] * ratio_H[:, None], rtol=1.0e-12)

    # N2, only N species, rescaled back to its value at t (N correction)
    k_N2 = inference.gas.species_index("N2")
    assert np.allclose(Y_corr[:, k_N2], Y_old[:, k_N2], rtol=1.0e-12)

    # Species without H and N unchanged (O), except O2 (O correction)
    others = [k for k in np.where(~has_H)[0] if k not in [inference.gas.species_index("O2"), k_N2]]
    assert np.array_equal(Y_corr[:, others], Y_new[:, others])


def test_wan_all_rich():
    inference = element_corrector("mech_H2.yaml")
    Y_old, Y_new = perturbed_states(inference.gas, "H2", phi=3.0)   # Rich burnt gases: Y_O2 ~ 7e-5

    k_O2 = inference.gas.species_index("O2")
    Y_corr = inference.enforce_elements_balance_wan_all(Y_old, Y_new)

    assert np.allclose(elements(inference, Y_corr), elements(inference, Y_old), rtol=1.0e-10, atol=1.0e-14)
    assert np.all(Y_corr >= 0.0)

    # O2 only changed by relative factors close to 1, while the O correction with O2 only makes it negative
    assert np.allclose(Y_corr[:, k_O2] / Y_new[:, k_O2], 1.0, atol=0.05)
    assert np.any(inference.enforce_elements_balance_wan(Y_old, Y_new)[:, k_O2] < 0.0)


def test_wan_all_single_element():
    # Imbalance of N only, carried by N2 (only N species): N2 rescaled by Y_N(t) / Y_N(t+dt) (Eq. (8)), nothing else
    inference = element_corrector("mech_H2.yaml")
    Y_old, _ = perturbed_states(inference.gas, "H2")
    k_N2 = inference.gas.species_index("N2")
    Y_new = Y_old.copy()
    Y_new[:, k_N2] *= 1.01

    Y_corr = inference.enforce_elements_balance_wan_all(Y_old, Y_new)

    assert np.allclose(Y_corr, Y_old, rtol=1.0e-12, atol=1.0e-16)


def test_hybrid_criteria():
    inference = element_corrector("mech_H2.yaml")
    Y_old, Y_new = perturbed_states(inference.gas, "H2", n=5)

    # Element error: relative change of the element mass fractions present (H, O, N in H2/air)
    e_e = inference.elements_error(Y_old, Y_new)
    Y_a_old, Y_a_new = elements(inference, Y_old), elements(inference, Y_new)
    present = Y_a_old[0] > 0.0
    assert np.allclose(e_e, np.max(np.abs(Y_a_new[:, present] - Y_a_old[:, present]) / Y_a_old[:, present], axis=1))
    assert np.allclose(inference.elements_error(Y_old, Y_old), 0.0)
    assert np.allclose(inference.elements_error(Y_old, inference.enforce_elements_balance(Y_old, Y_new)), 0.0, atol=1.0e-12)

    # Choice of CVODE
    e_sum = np.abs(Y_new.sum(axis=1) - 1.0)
    tol = np.median(e_sum)
    assert not use_cvode(e_sum, e_e, 0, tol).any()
    assert np.array_equal(use_cvode(e_sum, e_e, 1, tol), e_sum > tol)
    assert np.array_equal(use_cvode(e_sum, e_e, 2, np.median(e_e)), e_e > np.median(e_e))
