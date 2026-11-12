"""
Test 35: check of the inputs of the stochastic reactors (generate_stochastic_database).

Check: an unknown mixing_model raises a ValueError naming the valid models. The check is done by all MPI
ranks (DummyComm here), so that no rank waits for a rank that stopped.

Independent of the other tests.
"""
import os

import oyaml as yaml
import pytest

from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database

import DummyComm as dc


def test_unknown_mixing_model(tmp_path):
    os.chdir(tmp_path)

    # Only the mixing model is read before the check
    with open("dtb_params.yaml", "w") as file:
        yaml.dump({"mixing_model": "CURLL"}, file)

    with pytest.raises(ValueError, match="mixing_model must be one of"):
        generate_stochastic_database(dc.DummyComm())
