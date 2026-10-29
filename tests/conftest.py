"""
Fixtures shared by the tests.

The H2 stochastic reactors database of tests 01 to 04 is built once per test session, in a temporary
folder: these tests can run in any order, alone, or in several sessions at the same time.
"""
import pytest
from h2_stoch_database import generate_h2_stoch_database, resample_h2_stoch_database


@pytest.fixture(scope="session")
def h2_stoch_dtb(tmp_path_factory):
    """Run folder containing STOCH_DTB_TEST/solutions.h5 (see h2_stoch_database.py)."""

    folder = tmp_path_factory.mktemp("h2_stoch_dtb")
    generate_h2_stoch_database(folder)

    return folder


@pytest.fixture(scope="session")
def h2_resampled_dtb(h2_stoch_dtb):
    """Same run folder, also containing STOCH_DTB_TEST/solutions_resamp200K.h5 (see h2_stoch_database.py)."""

    resample_h2_stoch_database(h2_stoch_dtb)

    return h2_stoch_dtb
