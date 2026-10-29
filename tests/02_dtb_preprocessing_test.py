"""
Test 02: database undersampling.

The session fixture h2_resampled_dtb (see h2_stoch_database.py, conftest.py) applies the temperature
threshold and the HRR-based undersampling in (PC1, PC2) space to the database of test 01, and writes
STOCH_DTB_TEST/solutions_resamp200K.h5.
Check: histograms of the resampled database are close to the references in H2_resampled_TEST_histograms/.
"""
import os
import numpy as np
from misc import extract_h2_dtvar_dtb_histograms
from h2_stoch_database import tests_dir, resampled_file


def test_dtb_resampling(h2_resampled_dtb):
    os.chdir(h2_resampled_dtb)

    tol = np.array([35.71, 2.62, 11.83, 5.0, 28.87, 12.84, 10.97, 3.56, 2.57])
    assert np.max(extract_h2_dtvar_dtb_histograms(f"./STOCH_DTB_TEST/{resampled_file}", os.path.join(tests_dir, "H2_resampled_TEST_histograms"), tol)) < np.float64(0.0)
