"""
Test 12: composition and elements helpers of tools/utilities.py.

Check:
- Yk_from_phi_HC gives the same fuel/air mass fractions as Cantera, for fuels with C, H, O and N.
- get_molar_mass_atomic_matrix keeps only the elements present in the species, and gives the
  same elements mass fractions as Cantera.

Independent of the other tests.
"""
import os
import numpy as np
import cantera as ct
import pytest
import ai_reacting_flows.tools.utilities as utils

mech_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms")


@pytest.mark.parametrize("fuel, mech", [("H2", "mech_H2.yaml"), ("CH4", "mech_CH4_gri.yaml"), ("NH3", "mech_nh3.yaml")])
def test_Yk_from_phi(fuel, mech):
    gas = ct.Solution(os.path.join(mech_folder, mech))
    gas.TP = 300.0, 101325.0
    gas.set_equivalence_ratio(0.8, fuel, {"O2": 1.0, "N2": 3.76})

    compo = utils.Yk_from_phi_HC(fuel, 0.8)

    for spec in [fuel, "O2", "N2"]:
        assert np.isclose(compo[spec], gas[spec].Y[0], rtol=1.0e-4)


@pytest.mark.parametrize("with_N2_chemistry, elements", [(True, ["H", "O", "N"]), (False, ["H", "O"])])
def test_elements_matrix_H2(with_N2_chemistry, elements):
    gas = ct.Solution(os.path.join(mech_folder, "mech_H2.yaml"))
    gas.TP = 1500.0, 101325.0
    gas.set_equivalence_ratio(0.8, "H2", {"O2": 1.0, "N2": 3.76})
    gas.equilibrate("HP")

    A_element, names = utils.get_molar_mass_atomic_matrix(gas.species_names, with_N2_chemistry)
    assert names == elements

    # Elements mass fractions compared to Cantera (N2 removed from Y without N chemistry)
    Y = gas.Y if with_N2_chemistry else np.delete(gas.Y, gas.species_index("N2"))
    Y_elements = A_element @ Y
    for j, name in enumerate(names):
        assert np.isclose(Y_elements[j], gas.elemental_mass_fraction(name), rtol=1.0e-3)
