"""
Test 02: database undersampling.

The session fixture h2_resampled_dtb (see h2_stoch_database.py, conftest.py) applies the temperature
threshold and the HRR-based undersampling in (PC1, PC2) space to the database of test 01, and writes
STOCH_DTB_TEST/solutions_resamp200K.h5.
Check: histograms of the resampled database are close to the references in H2_resampled_TEST_histograms/.
The undersampling also works when nearly all points are kept (points of zero weight get a small probability).
"""
import os
import numpy as np
from misc import extract_h2_dtvar_dtb_histograms
from h2_stoch_database import tests_dir, resampled_file
from misc import fHRR
from ai_reacting_flows.databases_processing.database_processing import LearningDatabase


def test_dtb_resampling(h2_resampled_dtb):
    os.chdir(h2_resampled_dtb)

    tol = np.array([35.71, 2.62, 11.83, 5.0, 28.87, 12.84, 10.97, 3.56, 2.57])
    assert np.max(extract_h2_dtvar_dtb_histograms(f"./STOCH_DTB_TEST/{resampled_file}", os.path.join(tests_dir, "H2_resampled_TEST_histograms"), tol)) < np.float64(0.0)


def test_undersampling_most_points(h2_resampled_dtb):
    # Nearly all points kept: the points of zero sampling weight (outside the bin centers of the joint PDF)
    # must get a small non-zero probability
    os.chdir(h2_resampled_dtb)   # dtb_processing.yaml written by the fixture
    database = LearningDatabase()
    database.apply_temperature_threshold()
    n = database.X.shape[0]

    database.undersample_HRR("Temperature", "H2O", hrr_func=fHRR, keep_low_c=True, n_samples=n - 10, n_bins=100)
    assert database.X.shape[0] == n - 10
